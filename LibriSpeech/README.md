# LibriSpeech Speech Recognition Samples

This directory contains speech samples used to evaluate the effect of speech enhancement on automatic speech recognition (ASR), as reported in the HomoGenSE paper. The clean references are 500 utterances selected from the LibriSpeech `test-clean` set. Noisy speech was created by mixing the clean utterances with noise from the DNS Challenge noise dataset at five SNRs: −5, 0, 5, 10, and 15 dB. Audio is stored as FLAC.

## Contents

| Directory | Contents |
| --- | --- |
| `Clean/` | 500 clean reference utterances |
| `Noisy/` | Noisy versions grouped by SNR (`-5dB/`, `0dB/`, `5dB/`, `10dB/`, `15dB/`), 500 files per SNR |
| `HomoGenSE-DE/`, `HomoGenSE-NE/` | Enhanced outputs from the two HomoGenSE variants, grouped by SNR (`-5dB/`, `0dB/`, `5dB/`, `10dB/`, `15dB/`) |
| `CDiffuSE/`, `CTFSE/`, `DCCRN/`, `FlowSE-Freq/`, `FlowSE-Vocos/`, `FullSubNet/`, `MP-SENet/`, `NHS-SE+/`, `NSNet2/`, `SGMSE+/`, `StoRM/`, `TF-GridNet/` | Comparison-system outputs, grouped by the same SNR directory names |

Each SNR directory contains 500 files. All noisy and enhanced output folders use the same plain SNR names; specifically, the zero-SNR directory is named `0dB` (not `+0dB`). Within a condition, the utterance basename identifies the LibriSpeech recording and can be matched across the clean, noisy, and enhanced directories. For example, `2830-3980-0032.flac` is the same utterance across conditions.

The paper evaluates these samples with Whisper Tiny, Base, and Small, reporting word error rates at each SNR. It uses the official LibriSpeech transcripts as recognition references.
