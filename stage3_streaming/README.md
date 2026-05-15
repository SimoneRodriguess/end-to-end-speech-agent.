# end-to-end-speech-agent

A live speech-to-speech conversational agent built entirely in audio token space. No text. No ASR. No TTS in the inference loop. Pure audio token modeling.

You speak : your voice is compressed into discrete tokens via EnCodec : a custom transformer predicts response tokens : EnCodec decodes them back to audio : it plays through your speaker. The model never sees a single word.

## Stack
- Meta EnCodec 24kHz : audio to discrete token compression (8 codebooks, vocab 1027)
- TinyS2S : custom 13.7M param encoder-decoder transformer trained from scratch
- blended_skill_talk : real human conversational pairs synthesized via Coqui TTS
- Live VAD mic-to-speaker agent loop

## Latency
~3s end-to-end on RTX 3050 Laptop GPU
