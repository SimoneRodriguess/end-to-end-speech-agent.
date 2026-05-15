"""
stage1_codec/encode_decode.py
Verifies EnCodec encode/decode roundtrip.
Run: python encode_decode.py
"""
import sys
import numpy as np
import torch
import torchaudio
from pathlib import Path
from encodec import EncodecModel

OUT = Path("../outputs/stage1")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval().to(device)

    # Synthetic 2s sine wave at 24kHz
    sr = 24000
    t = torch.linspace(0, 2.0, 2 * sr)
    audio = (0.3 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0).unsqueeze(0)  # (1,1,T)
    audio = audio.to(device)
    print(f"Input audio shape: {audio.shape}")

    # Encode
    with torch.no_grad():
        encoded = model.encode(audio)
    codes = torch.cat([f[0] for f in encoded], dim=-1)
    print(f"Encoded codes shape: {codes.shape}  (batch, codebooks, frames)")

    # Decode
    with torch.no_grad():
        decoded = model.decode(encoded)
    print(f"Decoded audio shape: {decoded.shape}")

    # Save
    out_path = OUT / "reconstructed.wav"
    torchaudio.save(str(out_path), decoded.squeeze(0).cpu(), sr)
    print(f"Saved: {out_path}")
    print("Stage 1 OK.")

if __name__ == "__main__":
    main()
