"""
stage2_transformer/train.py
Trains TinyS2S on a dataset built by data.py.

Usage:
    python train.py train --data ../data/dataset.pt --size tiny --epochs 5 --batch 16 --name smoke
    python train.py train --data ../data/dataset_qa.pt --size small --epochs 60 --batch 8 --name qa_run1
    python train.py infer --ckpt ../runs/qa_run1/best.pt
"""
import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model import TinyS2S, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

RUNS = Path("../runs")
OUT  = Path("../outputs/stage2")
OUT.mkdir(parents=True, exist_ok=True)

SIZE_CFG = {
    "tiny":  dict(d_model=128, n_heads=2, n_layers=2),
    "small": dict(d_model=256, n_heads=4, n_layers=4),
    "base":  dict(d_model=512, n_heads=8, n_layers=6),
}


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, pairs, max_len=256):
        self.pairs   = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src = src[:, :self.max_len].long()  # (K, T)
        tgt = tgt[:, :self.max_len].long()
        return src, tgt


def collate(batch):
    """Pad variable-length sequences. Returns (B,T,K) tensors."""
    srcs, tgts = zip(*batch)
    K = srcs[0].shape[0]

    def pad(seqs):
        T = max(s.shape[1] for s in seqs)
        out = torch.full((len(seqs), K, T), PAD_TOKEN, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :, :s.shape[1]] = s
        return out.permute(0, 2, 1)  # (B, T, K)

    return pad(srcs), pad(tgts)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_dir = RUNS / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    saved = torch.load(args.data, map_location="cpu")
    pairs = saved["pairs"]
    print(f"Loaded {len(pairs)} pairs from {args.data}")

    dataset = PairDataset(pairs)
    val_n   = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate)

    # Model
    cfg = SIZE_CFG[args.size]
    model = TinyS2S(**cfg).to(device)
    print(f"Model: {args.size}  params={model.count_params()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            # Shift: input = tgt[:, :-1], label = tgt[:, 1:]
            tgt_in  = tgt[:, :-1, :]
            tgt_out = tgt

            # Prepend BOS frame to tgt_in
            bos = torch.full((src.size(0), 1, src.size(2)), BOS_TOKEN,
                             dtype=torch.long, device=device)
            tgt_in = torch.cat([bos, tgt_in], dim=1)

            logits = model(src, tgt_in)  # (B, T, K, vocab)
            B, T, K, V = logits.shape
            loss = criterion(
                logits.reshape(B * T * K, V),
                tgt_out.reshape(B * T * K)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / len(train_dl)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in  = tgt[:, :-1, :]
                tgt_out = tgt
                bos = torch.full((src.size(0), 1, src.size(2)), BOS_TOKEN,
                                 dtype=torch.long, device=device)
                tgt_in = torch.cat([bos, tgt_in], dim=1)
                logits = model(src, tgt_in)
                B, T, K, V = logits.shape
                val_loss += criterion(
                    logits.reshape(B * T * K, V),
                    tgt_out.reshape(B * T * K)
                ).item()
        avg_val = val_loss / len(val_dl)

        print(f"Epoch {epoch:3d} | train={avg_train:.4f} | val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": {**cfg, "n_codebooks": 8, "codebook_size": 1024, "max_seq_len": 256},
                "val_loss": best_val,
            }, ckpt_dir / "best.pt")
            print(f"  → saved best checkpoint (val={best_val:.4f})")

    print(f"Training done. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
@torch.no_grad()
def infer(args):
    import torchaudio
    from encodec import EncodecModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load codec
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(6.0)
    codec.eval().to(device)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt.get("config", {})
    model = TinyS2S(
        d_model   = cfg.get("d_model", 256),
        n_heads   = cfg.get("n_heads", 4),
        n_layers  = cfg.get("n_layers", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)

    # Dummy input: 1s sine wave
    sr = 24000
    import math, numpy as np
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = torch.from_numpy(0.3 * np.sin(2 * math.pi * 440 * t)).unsqueeze(0).unsqueeze(0).to(device)

    encoded = codec.encode(audio)
    codes_in = torch.cat([f.codes for f in encoded], dim=-1)  # (1, K, T)
    src = codes_in.permute(0, 2, 1)  # (1, T, K)

    # Autoregressive decode
    K = src.shape[2]
    gen = torch.full((1, 1, K), BOS_TOKEN, dtype=torch.long, device=device)
    for _ in range(200):
        logits = model(src, gen)
        next_tok = logits[:, -1, :, :].argmax(-1).unsqueeze(1)  # (1,1,K)
        gen = torch.cat([gen, next_tok], dim=1)
        if next_tok[0, 0, 0].item() == EOS_TOKEN:
            break

    codes_out = gen[:, 1:, :].permute(0, 2, 1)  # (1, K, T)
    decoded = codec.decode([(codes_out, None)])

    out_path = OUT / "stage2_response.wav"
    torchaudio.save(str(out_path), decoded.squeeze(0).cpu(), sr)
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("--data",   required=True)
    tr.add_argument("--size",   default="small", choices=["tiny", "small", "base"])
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch",  type=int, default=8)
    tr.add_argument("--lr",     type=float, default=1e-4)
    tr.add_argument("--name",   default="run1")

    inf = sub.add_parser("infer")
    inf.add_argument("--ckpt", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    else:
        parser.print_help()
