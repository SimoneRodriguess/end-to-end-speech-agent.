"""
stage3_streaming/agent.py
Live mic → EnCodec → S2S Transformer → EnCodec → speaker

Usage:
    python agent.py --ckpt ../runs/run1/best.pt
    python agent.py --ckpt ../runs/run1/best.pt --device cuda --threshold 0.02
"""

import sys
import os
import argparse
import time
import threading
from pathlib import Path

import numpy as np
import torch
import sounddevice as sd
from scipy.signal import resample_poly

# ------------------------------------------------------------------
# Add parent dirs to path so we can import stage1 / stage2 modules
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stage1_codec"))
sys.path.insert(0, str(ROOT / "stage2_transformer"))

from encodec import EncodecModel                    # pip install encodec
from encodec.utils import convert_audio
import torchaudio

# Import your model — adjust the class name if yours differs
from model import TinyS2S, BOS_TOKEN, EOS_TOKEN     # stage2_transformer/model.py


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SAMPLE_RATE_MIC   = 44100   # most mics default to this
SAMPLE_RATE_CODEC = 24000   # EnCodec 24k bandwidth model
CHUNK_SEC         = 0.1     # record in 100 ms chunks
CHUNK_SAMPLES     = int(SAMPLE_RATE_MIC * CHUNK_SEC)
MAX_RECORD_SEC    = 8       # cut off after this if user doesn't pause
MIN_SPEECH_SEC    = 0.3     # ignore taps shorter than this
SILENCE_THRESHOLD = 0.015   # RMS below this = silence (tune with --threshold)
SILENCE_CHUNKS    = 12      # ~1.2 s of silence triggers end-of-turn


# ------------------------------------------------------------------
# Load models once at startup
# ------------------------------------------------------------------
def load_models(ckpt_path: str, device: str):
    print("[agent] Loading EnCodec 24k …")
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(6.0)   # 8 codebooks
    codec.eval().to(device)

    print(f"[agent] Loading S2S checkpoint: {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both raw state-dict saves and wrapped saves
    cfg  = ckpt.get("config", {})
    sd_  = ckpt.get("model_state_dict", ckpt)

    model = TinyS2S(
        n_codebooks  = cfg.get("n_codebooks",  8),
        codebook_size= cfg.get("codebook_size", 1024),
        d_model      = cfg.get("d_model",       256),
        n_heads      = cfg.get("n_heads",       4),
        n_layers     = cfg.get("n_layers",      4),
        max_seq_len  = 512,
    )
    model.load_state_dict(sd_, strict=False)
    model.eval().to(device)
    print("[agent] Models ready.\n")
    return codec, model


# ------------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------------
def resample_to_codec(audio_np: np.ndarray) -> torch.Tensor:
    """Mic audio (SAMPLE_RATE_MIC, float32 numpy) → codec tensor (1, 1, T)."""
    if SAMPLE_RATE_MIC != SAMPLE_RATE_CODEC:
        from math import gcd
        g = gcd(SAMPLE_RATE_CODEC, SAMPLE_RATE_MIC)
        up, down = SAMPLE_RATE_CODEC // g, SAMPLE_RATE_MIC // g
        audio_np = resample_poly(audio_np, up, down).astype(np.float32)
    t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    return t


def encode_audio(codec, audio_t: torch.Tensor, device: str):
    """Return EnCodec codes: (1, n_codebooks, T_frames)."""
    audio_t = audio_t.to(device)
    with torch.no_grad():
        encoded = codec.encode(audio_t)
    # encoded is a list of EncodedFrame; concat along time
    codes = torch.cat([f[0] for f in encoded], dim=-1)  # (1, K, T)
    return codes


def decode_codes(codec, codes: torch.Tensor, device: str) -> np.ndarray:
    """EnCodec codes → float32 numpy waveform at SAMPLE_RATE_CODEC."""
    codes = codes.to(device)
    with torch.no_grad():
        from encodec.model import EncodedFrame
        frames = [(codes, None)]
        audio = codec.decode(frames)           # (1, 1, T)
    return audio.squeeze().cpu().numpy()


# ------------------------------------------------------------------
# Model inference: codes_in → codes_out (autoregressive)
# ------------------------------------------------------------------
@torch.no_grad()
def generate_response(model, codes_in: torch.Tensor, device: str,
                       max_new_frames: int = 200,
                       temperature: float = 0.8) -> torch.Tensor:
    """
    codes_in: (1, K, T_in)
    Returns codes_out: (1, K, T_out)

    The model predicts one frame (K codebook values) at a time.
    This uses teacher-forced input + autoregressive output (seq2seq style).
    Adjust to match your model's actual forward() signature.
    """
    model.eval()
    K = codes_in.shape[1]

    # Flatten input: (1, T_in, K)
    codes_in = codes_in[:, :, :512]
    src = codes_in.permute(0, 2, 1).to(device)  # (1, T, K)

    # Start with BOS
    gen = torch.full((1, 1, K), BOS_TOKEN, dtype=torch.long, device=device)

    for _ in range(max_new_frames):
        logits = model(src, gen)          # (1, T_gen, K, vocab)
        next_logits = logits[:, -1, :, :]  # (1, K, vocab)

        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(1, 1, K)
        else:
            next_tok = next_logits.argmax(-1).unsqueeze(1)  # (1, 1, K)

        gen = torch.cat([gen, next_tok], dim=1)

        # Stop on EOS in first codebook (heuristic)
        if (next_tok[0, 0, 0] == EOS_TOKEN).item():
            break

    # Remove BOS frame, convert to (1, K, T)
    gen = gen[:, 1:, :].permute(0, 2, 1)
    return gen  # (1, K, T_out)


# ------------------------------------------------------------------
# VAD recorder: blocks until one complete utterance is captured
# ------------------------------------------------------------------
def record_utterance(threshold: float, verbose: bool = True) -> np.ndarray:
    """
    Records from the default mic.
    Returns float32 numpy array at SAMPLE_RATE_MIC once silence is detected.
    """
    buffer    = []
    silent    = 0
    speaking  = False
    chunks_in = 0
    max_chunks = int(MAX_RECORD_SEC / CHUNK_SEC)

    if verbose:
        print("  [mic] Listening … (speak now)")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE_MIC,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
    )
    stream.start()

    try:
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk[:, 0]                   # mono
            rms = float(np.sqrt(np.mean(chunk**2)))

            if rms > threshold:
                speaking = True
                silent   = 0
                buffer.append(chunk)
                chunks_in += 1
            elif speaking:
                silent += 1
                buffer.append(chunk)              # include trailing silence
                if silent >= SILENCE_CHUNKS:
                    break
            # else: haven't started yet, skip

            if chunks_in >= max_chunks:
                if verbose:
                    print("  [mic] Max duration reached.")
                break
    finally:
        stream.stop()
        stream.close()

    audio = np.concatenate(buffer, axis=0) if buffer else np.zeros(1, np.float32)
    duration = len(audio) / SAMPLE_RATE_MIC

    if duration < MIN_SPEECH_SEC:
        return None  # too short, ignore

    if verbose:
        print(f"  [mic] Captured {duration:.2f}s of audio.")
    return audio


# ------------------------------------------------------------------
# Main conversation loop
# ------------------------------------------------------------------
def run_agent(ckpt_path: str,
              device: str = "cpu",
              threshold: float = SILENCE_THRESHOLD,
              temperature: float = 0.8):

    codec, model = load_models(ckpt_path, device)

    print("=" * 55)
    print("  Conversational agent ready.")
    print("  Speak into your mic. Ctrl+C to quit.")
    print("=" * 55 + "\n")

    turn = 0
    while True:
        turn += 1
        print(f"[Turn {turn}] Waiting for your voice …")

        # --- 1. Record utterance ---
        raw_audio = None
        while raw_audio is None:
            raw_audio = record_utterance(threshold=threshold)
            if raw_audio is None:
                print("  (too short, try again)")

        # --- 2. Resample + encode ---
        print("  [codec] Encoding …")
        audio_t  = resample_to_codec(raw_audio)
        codes_in = encode_audio(codec, audio_t, device)
        print(f"  [codec] Input codes: {codes_in.shape}  (1, codebooks, frames)")

        # --- 3. Run model ---
        print("  [model] Generating response …")
        t0 = time.time()
        codes_out = generate_response(model, codes_in, device,
                                       temperature=temperature)
        elapsed = time.time() - t0
        print(f"  [model] Output codes: {codes_out.shape}  ({elapsed:.2f}s)")

        # --- 4. Decode + play ---
        print("  [codec] Decoding response audio …")
        response_audio = decode_codes(codec, codes_out, device)

        print("  [speaker] Playing …")
        sd.play(response_audio, samplerate=SAMPLE_RATE_CODEC, blocking=True)
        print()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2S Conversational Agent")
    parser.add_argument("--ckpt",      required=True, help="Path to checkpoint .pt")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD,
                        help="RMS silence threshold (default 0.015). Increase in noisy rooms.")
    parser.add_argument("--temp",      type=float, default=0.8,
                        help="Sampling temperature (0 = greedy, 0.8 = default)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    try:
        run_agent(
            ckpt_path = args.ckpt,
            device    = args.device,
            threshold = args.threshold,
            temperature = args.temp,
        )
    except KeyboardInterrupt:
        print("\n[agent] Exiting.")
