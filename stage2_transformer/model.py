"""
stage2_transformer/model.py
TinyS2S: sequence-to-sequence transformer over EnCodec tokens.
Run standalone to verify forward pass: python model.py
"""
import torch
import torch.nn as nn
import math

BOS_TOKEN = 1024
EOS_TOKEN = 1025
PAD_TOKEN = 1026
VOCAB_SIZE = 1027  # 0-1023 codec + BOS/EOS/PAD


class TokenEmbedding(nn.Module):
    """Embeds all codebooks and sums them."""
    def __init__(self, n_codebooks, vocab_size, d_model):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(n_codebooks)
        ])

    def forward(self, x):
        # x: (B, T, K)
        out = sum(emb(x[:, :, i]) for i, emb in enumerate(self.embeddings))
        return out  # (B, T, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TinyS2S(nn.Module):
    def __init__(self, n_codebooks=8, codebook_size=1024, d_model=256,
                 n_heads=4, n_layers=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.d_model = d_model

        vocab = codebook_size + 3  # +BOS/EOS/PAD

        self.src_embed = TokenEmbedding(n_codebooks, vocab, d_model)
        self.tgt_embed = TokenEmbedding(n_codebooks, vocab, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        # One output head per codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab) for _ in range(n_codebooks)
        ])

    def forward(self, src, tgt):
        """
        src: (B, T_src, K)  input codes (question)
        tgt: (B, T_tgt, K)  target codes (answer, shifted right)
        Returns logits: (B, T_tgt, K, vocab)
        """
        T_tgt = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=src.device)

        src_e = self.pos_enc(self.src_embed(src))
        tgt_e = self.pos_enc(self.tgt_embed(tgt))

        out = self.transformer(src_e, tgt_e, tgt_mask=tgt_mask)  # (B, T_tgt, d_model)

        logits = torch.stack([h(out) for h in self.heads], dim=2)  # (B, T, K, vocab)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyS2S()
    print(f"TinyS2S params: {model.count_params() / 1e6:.1f}M")

    B, T_src, T_tgt, K = 2, 50, 30, 8
    src = torch.randint(0, 1024, (B, T_src, K))
    tgt = torch.randint(0, 1024, (B, T_tgt, K))

    logits = model(src, tgt)
    print(f"Logits shape: {logits.shape}  expected ({B}, {T_tgt}, {K}, 1027)")
    print("model.py OK.")
