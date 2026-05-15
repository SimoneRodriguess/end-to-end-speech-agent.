import sys, time, numpy as np, torch, sounddevice as sd
sys.path.insert(0, '../stage2_transformer')
sys.path.insert(0, '../stage1_codec')
from encodec import EncodecModel
from model import TinyS2S, BOS_TOKEN, EOS_TOKEN

device = 'cuda'
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)
codec.eval().to(device)
ckpt = torch.load('../runs/qa_run3/best.pt', map_location=device)
model = TinyS2S(d_model=256, n_heads=4, n_layers=4, max_seq_len=512)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval().to(device)

SR = 44100
THRESHOLD = 0.03
SILENCE_CHUNKS = 12
CHUNK = int(SR * 0.1)

print("Speak now...")
buf, silent, speaking = [], 0, False
stream = sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=CHUNK)
stream.start()
while True:
    chunk, _ = stream.read(CHUNK)
    chunk = chunk[:, 0]
    if np.sqrt(np.mean(chunk**2)) > THRESHOLD:
        speaking = True; silent = 0; buf.append(chunk)
    elif speaking:
        silent += 1; buf.append(chunk)
        if silent >= SILENCE_CHUNKS:
            break
stream.stop()

speech_end = time.time()
print(f"Speech captured. Processing...")

audio = np.concatenate(buf).astype(np.float32)
import torchaudio
t = torch.from_numpy(audio).unsqueeze(0)
t = torchaudio.functional.resample(t, SR, 24000).unsqueeze(0).to(device)
with torch.no_grad():
    encoded = codec.encode(t)
codes = torch.cat([f[0] for f in encoded], dim=-1)[:, :, :512]
src = codes.permute(0, 2, 1)
gen = torch.full((1,1,8), BOS_TOKEN, dtype=torch.long, device=device)
with torch.no_grad():
    for _ in range(50):
        logits = model(src, gen)
        next_tok = logits[:,-1,:,:].argmax(-1).unsqueeze(1)
        gen = torch.cat([gen, next_tok], dim=1)
        if next_tok[0,0,0].item() == EOS_TOKEN:
            break
codes_out = gen[:,1:,:].permute(0,2,1)
with torch.no_grad():
    response = codec.decode([(codes_out, None)])

speech_start = time.time()
latency = speech_start - speech_end
print(f"\nLatency: {latency*1000:.0f} ms")
sd.play(response.squeeze().cpu().numpy(), samplerate=24000, blocking=True)
