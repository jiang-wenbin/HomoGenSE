# VCTK+DEMAND Speech Enhancement Samples

This directory contains paired speech samples and speech-enhancement outputs used for the VCTK+DEMAND evaluation in the HomoGenSE paper.

## Contents

Each top-level subdirectory contains FLAC audio. For the flat directories, files with the same basename refer to the same utterance across the clean reference, noisy input, and enhancement systems.

| Directory | Contents |
| --- | --- |
| `clean/` | Clean reference speech (824 files) |
| `noisy/` | Noisy input speech (824 files) |
| `HomoGenSE-DE/`, `HomoGenSE-NE/` | HomoGenSE outputs; DE and NE variants (824 files each) |
| `HomoGenSE-DE_NFE2/`, `HomoGenSE-DE_NFE5/` | HomoGenSE-DE outputs using 2 and 5 function evaluations (824 files each) |
| `CDiffuSE/`, `CTFSE/`, `DCCRN/`, `FlowSE-Freq/`, `FlowSE-Vocos/`, `MP-SENet/`, `NHS-SE+/`, `OMLSA/`, `SGMSE+/`, `StoRM/`, `TF-GridNet/` | Outputs from comparison systems (824 files each) |

The filenames follow the VCTK speaker/utterance convention (for example, `p232_258.flac`). Match samples by basename when comparing systems. The HomoGenSE NFE variants allow comparison of different inference budgets; the paper uses NFE = 1 by default and also reports NFE = 2 and 5.
