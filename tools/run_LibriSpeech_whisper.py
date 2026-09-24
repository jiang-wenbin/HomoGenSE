"""Run OpenAI Whisper ASR on every SE method under a data root and aggregate
the WER results into an xlsx workbook.

Audio is read from <data-root>/..., references from --transcript, per-condition
CSVs are checkpoints under --whisper-dir, and the final matrices go to --out.

Handles the layouts used in this repo:
  * flat dir of flacs (e.g. clean/)           -> one condition "Clean"
  * top-level noisy_<SNR>/ dirs               -> method "Noisy"
  * dir with <SNR> subdirs                    -> method/<SNR>
  * dir with est_<SNR> subdirs (HomoGenSE-*)  -> method/est_<SNR>

Usage (from the repo root):
    python tools/run_LibriSpeech_whisper.py
    python tools/run_LibriSpeech_whisper.py --data-root LibriSpeech \
        --transcript exp/LibriSpeech/txt/transcript.txt \
        --whisper-dir exp/LibriSpeech_check/whisper \
        --out exp/LibriSpeech_check/Whisper.xlsx
"""

import argparse
import re
from pathlib import Path

import jiwer
import pandas as pd
from omegaconf import OmegaConf

import LibriSpeech

DEFAULT_SNRS = ["-5dB", "0dB", "5dB", "10dB", "15dB"]
SNR_RE = re.compile(r"^(?:est_|noisy_)?(-?\d+dB)$", re.IGNORECASE)


def snr_of(name):
    match = SNR_RE.match(name)
    return match.group(1) if match else None


def snr_sort_key(snr):
    return (DEFAULT_SNRS.index(snr) if snr in DEFAULT_SNRS else len(DEFAULT_SNRS), snr)


def detect_ext(directory):
    """Extension of the audio files in a condition dir ('flac' or 'wav')."""
    for p in sorted(Path(directory).iterdir()):
        if p.is_file() and p.suffix.lower() in (".flac", ".wav"):
            return p.suffix.lower().lstrip(".")
    return "flac"


def discover_plan(data_root):
    """[(method, subset, snr, wav_ext), ...] with method/SNR names normalized."""
    root = Path(data_root)
    plan = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        sub_snrs = {p.name: snr_of(p.name)
                    for p in sorted(entry.iterdir()) if p.is_dir()}
        sub_snrs = {s: v for s, v in sub_snrs.items() if v}
        if sub_snrs:                      # method dir with SNR (or est_SNR) subdirs
            for sub in sorted(sub_snrs, key=lambda s: snr_sort_key(sub_snrs[s])):
                plan.append((entry.name, f"{entry.name}/{sub}", sub_snrs[sub],
                             detect_ext(entry.joinpath(sub))))
            continue
        snr = snr_of(entry.name)
        if snr:                           # top-level noisy_<SNR> dir
            plan.append(("Noisy", entry.name, snr, detect_ext(entry)))
            continue
        ext = detect_ext(entry)
        if any(p.suffix == f".{ext}" for p in entry.iterdir() if p.is_file()):
            method = "Clean" if entry.name.lower() == "clean" else entry.name
            plan.append((method, entry.name, "Clean", ext))
    return plan


def order_methods(plan):
    methods = {m for m, _, _, _ in plan}
    return sorted(methods, key=lambda m: (m != "Clean", m != "Noisy", m))


def order_snrs(plan):
    snrs = {s for _, _, s, _ in plan if s != "Clean"}
    return sorted(snrs, key=snr_sort_key)


def subsets_of(plan, method):
    return [sub for m, sub, _snr, _ext in plan if m == method]


def run_asr(model, method, subsets, batch_size, data_root, transcript,
            whisper_dir, wav_ext):
    conf = OmegaConf.create({
        "cmd": "ASR",
        "model": model,
        "batch_size": batch_size,
        "data_root": data_root,
        "wav_scp": transcript,  # absolute path: works for every condition dir
        "wav_ext": wav_ext,
        "tag": f"{model}_{method}",
        "subset": ",".join(subsets),
        "result_path": whisper_dir,
    })
    print(f"==== {model} / {method} ====", flush=True)
    LibriSpeech.ASR(conf)


def collect(model, plan, n_ref, whisper_dir):
    records, details = [], []
    for method, sub, snr, _ext in plan:
        csv_path = Path(whisper_dir).joinpath(
            f"{model}_{method}_{sub.replace('/', '_')}.csv")
        if not csv_path.exists():
            continue
        data = pd.read_csv(csv_path, keep_default_na=False)
        if len(data) != n_ref:
            print(f"skip incomplete {csv_path} ({len(data)}/{n_ref})")
            continue
        # keep_default_na=False + str(): empty whisper hypotheses must stay
        # "" instead of NaN, otherwise jiwer rejects the float
        wer = jiwer.wer([str(x) for x in data["reference_clean"]],
                        [str(x) for x in data["hypothesis_clean"]]) * 100
        records.append(dict(method=method, snr=snr, wer=wer))
        details.append(pd.DataFrame({
            "model": model, "method": method, "snr": snr,
            "wav_id": data["wav_id"],
            "reference": data["reference"],
            "hypothesis": data["hypothesis"],
        }))
    return records, details


def write_xlsx(models_done, plan, n_ref, whisper_dir, out):
    methods, snrs = order_methods(plan), order_snrs(plan)
    sheets, frames = [], []
    for model in models_done:
        records, details = collect(model, plan, n_ref, whisper_dir)
        if records:
            matrix = (pd.DataFrame(records)
                      .pivot(index="method", columns="snr", values="wer")
                      .reindex(index=methods, columns=["Clean"] + snrs)
                      .round(2))
            sheets.append((f"WER_{model}", matrix))
            frames.extend(details)
    if not sheets:
        print(f"no completed results yet, skip writing {out}")
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        for name, matrix in sheets:
            matrix.to_excel(writer, sheet_name=name)
        if frames:
            pd.concat(frames, ignore_index=True).to_excel(
                writer, sheet_name="Detail", index=False)
    print(f"wrote {out} (models: {', '.join(models_done)})", flush=True)


def main(args):
    plan = discover_plan(args.data_root)
    if args.snrs:
        plan = [p for p in plan if p[2] in args.snrs.split(",")]
    if args.methods:
        plan = [p for p in plan if p[0] in args.methods.split(",")]
    methods = order_methods(plan)
    models = args.models.split(",")
    transcript = str(Path(args.transcript).resolve())  # absolute: joined per condition dir
    n_ref = sum(1 for line in open(transcript) if line.strip())
    print(f"model(s): {models}\ndata-root: {args.data_root}\n"
          f"methods ({len(methods)}): {methods}\n"
          f"snrs: {['Clean'] + order_snrs(plan)}", flush=True)
    for i, model in enumerate(models):
        for method in methods:
            entries = [p for p in plan if p[0] == method]
            # a method's condition dirs may mix audio extensions (flac/wav)
            for ext in dict.fromkeys(p[3] for p in entries):
                subsets = [p[1] for p in entries if p[3] == ext]
                run_asr(model, method, subsets, args.batch_size,
                        args.data_root, transcript, args.whisper_dir, ext)
        write_xlsx(models[:i + 1], plan, n_ref, args.whisper_dir, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="tiny.en,base.en,small.en",
                        help="comma list of whisper models, run in order")
    parser.add_argument("--methods", default=None,
                        help="comma list of methods; default: all dirs under --data-root")
    parser.add_argument("--snrs", default=None,
                        help="comma list of SNR labels; default: -5dB,0dB,5dB,10dB,15dB")
    parser.add_argument("--data-root", default="exp/LibriSpeech",
                        help="root holding the method dirs")
    parser.add_argument("--transcript", default="exp/LibriSpeech/txt/transcript.txt",
                        help="wav_id <space> text reference file")
    parser.add_argument("--whisper-dir", default="exp/LibriSpeech/whisper",
                        help="checkpoint dir for the per-condition CSVs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="exp/LibriSpeech/Whisper.xlsx")
    main(parser.parse_args())
