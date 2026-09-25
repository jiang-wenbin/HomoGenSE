# DNS Challenge Speech Enhancement Samples

This directory contains speech-enhancement evaluation samples for the DNS Challenge (DNS2021) experiments reported in the HomoGenSE paper. Audio files are provided as FLAC.

## Contents

The enhancement-method directories each contain 600 output files organized by evaluation condition. The reference `test_set/` contains 900 files arranged by condition and clean/noisy reference type.

| Directory | Contents |
| --- | --- |
| `test_set/` | DNS Challenge test references and inputs (`no_reverb/`, `with_reverb/`, and `real_recordings/` condition trees) |
| `HomoGenSE-DE/`, `HomoGenSE-NE/` | HomoGenSE outputs |
| `CDiffuSE/`, `CTFSE/`, `DCCRN/`, `FlowSE-Freq/`, `FlowSE-Vocos/`, `FullSubNet/`, `MP-SENet/`, `MP-SENet_retrain/`, `NHS-SE+/`, `NSNet2/`, `RE-USE/`, `SEMamba/`, `SGMSE+/`, `StoRM/`, `TF-GridNet/` | Outputs from comparison systems |

System outputs are grouped into condition subdirectories: `no_reverb/`, `with_reverb/`, and `real_recordings/`. Simulated references are further separated into `clean/` and `noisy/` under `test_set/`; the real-recording condition has no clean reference. Corresponding files use matching names (for example, `no_reverb_283.flac`); use the condition and filename together to identify the sample. The paper evaluates simulated speech both without and with reverberation, as well as real recordings.
