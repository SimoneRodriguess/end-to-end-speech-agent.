"""
stage2_transformer/data.py
Builds a paired (input_codes, output_codes) dataset for S2S training.

Sources:
    synthetic   — sine-wave pairs (smoke test only, no real speech)
    ljspeech    — single LJSpeech speaker, consecutive sentence pairs
                  (learns prosody/timing but NOT Q→A semantics)
    qa_synth    — DailyDialog text Q&A synthesized with Coqui TTS
                  (recommended: learns real conversational structure)

Usage:
    # Quickest smoke-test (no downloads):
    python data.py --source synthetic --n 300 --output ../data/dataset.pt

    # Recommended for a working conversational agent:
    python data.py --source qa_synth --n 2000 --output ../data/dataset_qa.pt

    # Full dataset (~25k pairs, takes ~1 hr):
    python data.py --source qa_synth --n 0 --output ../data/dataset_qa_full.pt

Requirements for qa_synth:
    pip install TTS datasets soundfile
    (Coqui TTS downloads ~100 MB model on first run)
"""

import sys
import os
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Shared codec loader (keep one instance across the module)
# ------------------------------------------------------------------
_codec = None

def get_codec(device="cpu"):
    global _codec
    if _codec is None:
        from encodec import EncodecModel
        _codec = EncodecModel.encodec_model_24khz()
        _codec.set_target_bandwidth(6.0)   # 8 codebooks, 6 kbps
        _codec.eval().to(device)
    return _codec


def audio_to_codes(audio_np: np.ndarray, sr: int, device="cpu") -> torch.Tensor:
    """
    audio_np : float32 numpy (T,)
    sr       : source sample rate (will be resampled to 24 kHz)
    Returns  : (1, K, T_frames) int64 tensor
    """
    import torchaudio
    codec = get_codec(device)

    t = torch.from_numpy(audio_np).float().unsqueeze(0)     # (1, T)
    if sr != 24000:
        t = torchaudio.functional.resample(t, sr, 24000)
    t = t.unsqueeze(0).to(device)                            # (1, 1, T)

    with torch.no_grad():
        encoded = codec.encode(t)
    codes = torch.cat([f[0] for f in encoded], dim=-1)   # (1, K, T)
    return codes.cpu()


# ------------------------------------------------------------------
# Source: synthetic
# ------------------------------------------------------------------
def make_synthetic_pairs(n: int) -> list:
    """
    Sine-wave pairs. Useful only to verify shapes and that training runs.
    No conversational semantics whatsoever.
    """
    pairs = []
    sr = 24000
    for _ in tqdm(range(n), desc="Synthetic"):
        dur_in  = random.uniform(1.0, 3.0)
        dur_out = random.uniform(1.0, 3.0)
        f_in  = random.uniform(200, 800)
        f_out = random.uniform(200, 800)

        t_in  = np.linspace(0, dur_in,  int(sr * dur_in),  dtype=np.float32)
        t_out = np.linspace(0, dur_out, int(sr * dur_out), dtype=np.float32)

        a_in  = (0.3 * np.sin(2 * math.pi * f_in  * t_in)).astype(np.float32)
        a_out = (0.3 * np.sin(2 * math.pi * f_out * t_out)).astype(np.float32)

        c_in  = audio_to_codes(a_in,  sr)
        c_out = audio_to_codes(a_out, sr)
        pairs.append((c_in[0], c_out[0]))   # strip batch dim → (K, T)
    return pairs


# ------------------------------------------------------------------
# Source: ljspeech
# ------------------------------------------------------------------
def make_ljspeech_pairs(n: int, lj_root: str) -> list:
    """
    Consecutive LJSpeech sentence pairs.
    Learns prosody and timing but NOT question→answer semantics.

    lj_root: path to LJSpeech-1.1/ (contains wavs/ and metadata.csv)
    Download: https://keithito.com/LJ-Speech-Dataset/
    """
    import torchaudio
    root = Path(lj_root)
    wav_dir = root / "wavs"
    meta = root / "metadata.csv"

    lines = meta.read_text().strip().split("\n")
    ids = [l.split("|")[0] for l in lines]
    random.shuffle(ids)

    pairs = []
    limit = n if n > 0 else len(ids) - 1

    for i in tqdm(range(min(limit, len(ids) - 1)), desc="LJSpeech"):
        try:
            p_in  = wav_dir / f"{ids[i]}.wav"
            p_out = wav_dir / f"{ids[i+1]}.wav"
            a_in,  sr_in  = torchaudio.load(str(p_in))
            a_out, sr_out = torchaudio.load(str(p_out))

            c_in  = audio_to_codes(a_in[0].numpy(),  sr_in)
            c_out = audio_to_codes(a_out[0].numpy(), sr_out)
            pairs.append((c_in[0], c_out[0]))
        except Exception as e:
            print(f"  skip {ids[i]}: {e}")
    return pairs


# ------------------------------------------------------------------
# Source: qa_synth   ← the one you want for a conversational agent
# ------------------------------------------------------------------
def make_qa_synth_pairs(n: int, device: str = "cpu") -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        raise ImportError("pip install TTS")
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("pip install soundfile")
    import tempfile

    print("[data] Loading blended_skill_talk dataset ...")
    ds = load_dataset("blended_skill_talk", split="train")

    text_pairs = []
    for item in ds:
        turns = item["previous_utterance"]
        for i in range(len(turns) - 1):
            q = turns[i].strip()
            a = turns[i + 1].strip()
            if 5 < len(q) < 200 and 5 < len(a) < 200:
                text_pairs.append((q, a))

    random.shuffle(text_pairs)
    limit = n if n > 0 else len(text_pairs)
    text_pairs = text_pairs[:limit]
    print(f"[data] {len(text_pairs)} text pairs extracted.")

    print("[data] Loading Coqui TTS model ...")
    tts_model = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")

    pairs = []
    failed = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for q_text, a_text in tqdm(text_pairs, desc="Synthesizing pairs"):
            try:
                q_path = os.path.join(tmpdir, "q.wav")
                a_path = os.path.join(tmpdir, "a.wav")
                tts_model.tts_to_file(text=q_text, file_path=q_path)
                tts_model.tts_to_file(text=a_text, file_path=a_path)
                q_audio, q_sr = sf.read(q_path, dtype="float32")
                a_audio, a_sr = sf.read(a_path, dtype="float32")
                if q_audio.ndim == 2: q_audio = q_audio.mean(axis=1)
                if a_audio.ndim == 2: a_audio = a_audio.mean(axis=1)
                c_in  = audio_to_codes(q_audio, q_sr, device)
                c_out = audio_to_codes(a_audio, a_sr, device)
                pairs.append((c_in[0], c_out[0]))
            except Exception as e:
                failed += 1
                if failed < 5:
                    print(f"  [warn] skipped: {e}")

    print(f"[data] Built {len(pairs)} pairs. ({failed} skipped)")
    return pairs
# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------
class SpeechPairDataset(Dataset):
    """
    Each sample: dict with 'input' and 'target' — both (K, T) int64 tensors.
    Sequences are truncated to max_len frames (not padded here; train.py pads).
    """

    def __init__(self, pairs: list, max_len: int = 256):
        self.pairs   = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src = src[:, :self.max_len]
        tgt = tgt[:, :self.max_len]
        return {"input": src.long(), "target": tgt.long()}


def print_stats(pairs: list):
    in_lens  = [p[0].shape[-1] for p in pairs]
    out_lens = [p[1].shape[-1] for p in pairs]
    print(f"  Pairs         : {len(pairs)}")
    print(f"  Input  frames : min={min(in_lens)}, max={max(in_lens)}, mean={sum(in_lens)/len(in_lens):.1f}")
    print(f"  Output frames : min={min(out_lens)}, max={max(out_lens)}, mean={sum(out_lens)/len(out_lens):.1f}")
    print(f"  Codebooks     : {pairs[0][0].shape[0]}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="qa_synth",
                        choices=["synthetic", "ljspeech", "qa_synth"])
    parser.add_argument("--n",       type=int, default=2000,
                        help="Pairs to generate (0 = all available)")
    parser.add_argument("--output",  default="../data/dataset.pt")
    parser.add_argument("--lj_root", default="../data/LJSpeech-1.1",
                        help="LJSpeech root dir (only for --source ljspeech)")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max codec frames per sequence (truncated in Dataset)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        pairs = make_synthetic_pairs(args.n if args.n > 0 else 300)
    elif args.source == "ljspeech":
        pairs = make_ljspeech_pairs(args.n, args.lj_root)
    elif args.source == "qa_synth":
        pairs = make_qa_synth_pairs(args.n, args.device)

    print("\nDataset stats:")
    print_stats(pairs)

    dataset = SpeechPairDataset(pairs, max_len=args.max_len)
    torch.save({"pairs": pairs, "max_len": args.max_len, "source": args.source},
               args.output)
    print(f"\nSaved → {args.output}")
