"""Build the Section 5(d) results table from fever_run_200.csv.

Emits per-class precision / recall / F1 / support and macro-F1 for BOTH
argmax(T, I, F) (NeutroRAG) and the raw argmax(e, n, c) NLI baseline, side by
side, plus Wald confidence intervals and an exact McNemar test on the paired
predictions -- so the numbers can live in the report body as a table rather
than inside a screenshot (memo F18, F17).

Reads only the committed CSV; no model is loaded and no network is used.

    python experiments/make_results_table.py
    python experiments/make_results_table.py --csv fever_run_200.csv --outdir results
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

CHANNELS = ("T", "I", "F")
CHANNEL_GLOSS = {"T": "T (SUPPORTS)", "I": "I (NOT ENOUGH INFO)", "F": "F (REFUTES)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="fever_run_200.csv")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


def raw_channel(row: dict) -> str:
    """argmax over the NLI head's own distribution: e->T, n->I, c->F."""
    vals = {"T": float(row["e"]), "I": float(row["n"]), "F": float(row["c"])}
    return max(vals, key=vals.get)


def confusion(pairs):
    m = {g: {p: 0 for p in CHANNELS} for g in CHANNELS}
    for gold, pred in pairs:
        m[gold][pred] += 1
    return m


def per_class(m):
    """Return {channel: (precision, recall, f1, support)}."""
    out = {}
    for ch in CHANNELS:
        tp = m[ch][ch]
        fp = sum(m[g][ch] for g in CHANNELS if g != ch)
        fn = sum(m[ch][p] for p in CHANNELS if p != ch)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[ch] = (prec, rec, f1, tp + fn)
    return out


def accuracy(pairs) -> float:
    return sum(g == p for g, p in pairs) / len(pairs)


def wald_ci(acc: float, n: int):
    se = math.sqrt(acc * (1 - acc) / n)
    return se, acc - 1.96 * se, acc + 1.96 * se


def mcnemar_exact(pairs_a, pairs_b):
    """Two-sided exact (binomial) McNemar on paired predictions.

    b = A right where B wrong; c = B right where A wrong.
    """
    b = sum(1 for (ga, pa), (gb, pb) in zip(pairs_a, pairs_b) if ga == pa and gb != pb)
    c = sum(1 for (ga, pa), (gb, pb) in zip(pairs_a, pairs_b) if gb == pb and ga != pa)
    m = b + c
    if m == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    return b, c, min(1.0, 2 * tail)


def confusion_md(m, title: str) -> str:
    lines = [f"**{title}**", "",
             "| gold \\ predicted | T | I | F | support | recall |",
             "|---|---:|---:|---:|---:|---:|"]
    for g in CHANNELS:
        sup = sum(m[g].values())
        rec = m[g][g] / sup if sup else 0.0
        cells = " | ".join(str(m[g][p]) for p in CHANNELS)
        lines.append(f"| **{g}** | {cells} | {sup} | {rec:.3f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    n = len(rows)
    commit = git_commit()

    neutro = [(r["gold_channel"], r["predicted_channel"]) for r in rows]
    raw = [(r["gold_channel"], raw_channel(r)) for r in rows]

    m_neutro, m_raw = confusion(neutro), confusion(raw)
    pc_neutro, pc_raw = per_class(m_neutro), per_class(m_raw)
    acc_neutro, acc_raw = accuracy(neutro), accuracy(raw)
    macro_neutro = sum(v[2] for v in pc_neutro.values()) / len(CHANNELS)
    macro_raw = sum(v[2] for v in pc_raw.values()) / len(CHANNELS)
    se_n, lo_n, hi_n = wald_ci(acc_neutro, n)
    se_r, lo_r, hi_r = wald_ci(acc_raw, n)
    b, c, p = mcnemar_exact(neutro, raw)
    n_diff = sum(1 for a, r_ in zip(neutro, raw) if a[1] != r_[1])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- CSV -------------------------------------------------------------
    csv_path = outdir / "fever_200_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "class", "precision", "recall", "f1", "support"])
        for name, pc in (("neutrorag_argmax_TIF", pc_neutro), ("raw_nli_argmax_enc", pc_raw)):
            for ch in CHANNELS:
                prec, rec, f1, sup = pc[ch]
                w.writerow([name, ch, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", sup])
        w.writerow([])
        w.writerow(["system", "metric", "value"])
        for name, acc, macro, se, lo, hi in (
            ("neutrorag_argmax_TIF", acc_neutro, macro_neutro, se_n, lo_n, hi_n),
            ("raw_nli_argmax_enc", acc_raw, macro_raw, se_r, lo_r, hi_r),
        ):
            w.writerow([name, "accuracy", f"{acc:.4f}"])
            w.writerow([name, "macro_f1", f"{macro:.4f}"])
            w.writerow([name, "wald_se", f"{se:.4f}"])
            w.writerow([name, "ci95_low", f"{lo:.4f}"])
            w.writerow([name, "ci95_high", f"{hi:.4f}"])
        w.writerow(["paired", "mcnemar_b_neutro_only_right", b])
        w.writerow(["paired", "mcnemar_c_raw_only_right", c])
        w.writerow(["paired", "mcnemar_exact_p", f"{p:.4f}"])
        w.writerow(["paired", "rows_where_predictions_differ", n_diff])

    # ---- Markdown --------------------------------------------------------
    md = []
    md.append("# FEVER failure-type attribution: NeutroRAG vs. raw NLI baseline")
    md.append("")
    md.append(f"N = {n} claims, `copenlu/fever_gold_evidence` validation split, scored with "
              "`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.")
    md.append("")
    md.append(f"Source: `{args.csv}` | generated by `python experiments/make_results_table.py` "
              f"| commit `{commit}`")
    md.append("")
    md.append("Both systems use the **identical** NLI calls. The baseline takes `argmax` over "
              "the NLI head's own distribution (e->T, n->I, c->F); NeutroRAG takes `argmax` over "
              "the (T, I, F) triple built from it.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| system | accuracy | 95% CI (Wald) | macro-F1 |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| NeutroRAG `argmax(T,I,F)` | {acc_neutro:.4f} | [{lo_n:.3f}, {hi_n:.3f}] | {macro_neutro:.4f} |")
    md.append(f"| raw NLI `argmax(e,n,c)` | {acc_raw:.4f} | [{lo_r:.3f}, {hi_r:.3f}] | {macro_raw:.4f} |")
    md.append(f"| **difference** | **{acc_neutro - acc_raw:+.4f}** | | **{macro_neutro - macro_raw:+.4f}** |")
    md.append("")
    md.append(f"Predictions differ on **{n_diff} of {n}** rows. Exact McNemar on the paired "
              f"predictions: b = {b} (NeutroRAG only right), c = {c} (baseline only right), "
              f"**p = {p:.4f}**.")
    md.append("")
    md.append("## Per-class")
    md.append("")
    md.append("| class | \\| | NeutroRAG P | R | F1 | \\| | raw NLI P | R | F1 | \\| | support |")
    md.append("|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|")
    for ch in CHANNELS:
        pn, rn, fn_, sup = pc_neutro[ch]
        pr, rr, fr, _ = pc_raw[ch]
        md.append(f"| {CHANNEL_GLOSS[ch]} | | {pn:.3f} | {rn:.3f} | {fn_:.3f} | | "
                  f"{pr:.3f} | {rr:.3f} | {fr:.3f} | | {sup} |")
    md.append(f"| **macro-F1** | | | | **{macro_neutro:.4f}** | | | | **{macro_raw:.4f}** | | {n} |")
    md.append("")
    md.append("## Confusion matrices")
    md.append("")
    md.append(confusion_md(m_neutro, "NeutroRAG -- argmax(T, I, F)"))
    md.append("")
    md.append(confusion_md(m_raw, "Baseline -- argmax(e, n, c)"))
    md.append("")
    md_path = outdir / "fever_200_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print()
    print(f"NeutroRAG argmax(T,I,F): acc={acc_neutro:.4f} +/-{se_n:.4f} macro-F1={macro_neutro:.4f}")
    print(f"raw NLI   argmax(e,n,c): acc={acc_raw:.4f} +/-{se_r:.4f} macro-F1={macro_raw:.4f}")
    print(f"difference             : {acc_neutro - acc_raw:+.4f} accuracy, "
          f"{macro_neutro - macro_raw:+.4f} macro-F1")
    print(f"McNemar exact          : b={b} c={c} p={p:.4f} (differ on {n_diff}/{n} rows)")


if __name__ == "__main__":
    main()
