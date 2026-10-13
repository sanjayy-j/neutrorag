"""NeutroRAG end-to-end demo (offline, mock NLI).

Runs three things:
  1. the FEVER -> (T, I, F) proof-of-concept (does the dominant channel match?),
  2. the five-way decision policy on illustrative cases,
  3. the fuzzy-collapse ablation (full neutrosophic vs. I+F merged).

    python demo.py

NOTE: numbers below come from the deterministic MOCK NLI backend. Swap in
`HFNLIBackend` and real FEVER for paper-grade results.
"""
from neutrorag import Evidence, NeutroPolicy, NLIRetrievalScorer, MockNLIBackend
from neutrorag.eval import (
    failure_type_attribution, format_confusion, fuzzy_collapse_ablation,
)
from neutrorag.fever import load_sample, score_examples


def hr(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    policy = NeutroPolicy()

    print("=" * 68)
    print("NeutroRAG demo -- ILLUSTRATIVE PLUMBING, NOT EXPERIMENTAL RESULTS")
    print("=" * 68)
    print("Backend : MockNLIBackend -- a lexical-overlap heuristic, NOT an NLI model.")
    print("Data    : 12 hand-written examples embedded in fever.py.")
    print("The figures below show that the pipeline runs. They are NOT research")
    print("results and must not be quoted as any. The project's real numbers")
    print("live in RESULTS.md, with the commands that produced them.")
    print("=" * 68)

    # 1) FEVER proof-of-concept -------------------------------------------------
    hr("1. FEVER -> (T, I, F) premise check")
    examples = load_sample()
    tifs, gold = score_examples(examples, scorer)
    attr = failure_type_attribution(tifs, gold)
    print(f"[illustrative, mock NLI, n=12] dominant channel vs. gold: "
          f"{attr['accuracy']:.3f}")
    print("  ^ a plumbing check on 12 toy examples -- NOT the FEVER result.")
    print("    The real FEVER number is in RESULTS.md and is RETRACTED as a")
    print("    NeutroRAG result: a raw argmax(e,n,c) baseline scores higher.")
    print()
    print(format_confusion(attr["confusion"]))
    print("\nPer-example dominant channel:")
    for ex, tif in zip(examples, tifs):
        ok = "OK " if tif.dominant == ex.gold_channel else "MISS"
        print(f"  [{ok}] gold={ex.gold_channel}  {tif}  :: {ex.claim[:52]}")

    # 2) Decision policy on illustrative cases ---------------------------------
    hr("2. Five-way decision policy")
    cases = {
        "supported": [Evidence("The heart pumps blood through the body.", 0.9)],
        "contradicted": [Evidence("Sharks are not mammals they are fish.", 0.9)],
        "off-topic (missing)": [Evidence("The region has a temperate climate.", 0.4)],
        "conflicting sources": [
            Evidence("The deadline is Friday this week.", 0.9),
            Evidence("The deadline is not Friday it was moved.", 0.9),
        ],
        "no evidence": [],
    }
    claims = {
        "supported": "The heart pumps blood.",
        "contradicted": "Sharks are mammals.",
        "off-topic (missing)": "The mayor won an award in 1998.",
        "conflicting sources": "The deadline is Friday.",
        "no evidence": "An unverifiable private fact.",
    }

    def expand_fn(claim, evidence, step):
        return evidence  # demo: expansion finds nothing new

    for name, ev in cases.items():
        d = policy.run(claims[name], ev, scorer.score, expand_fn)
        print(f"  {name:22s} -> {d.action.value:22s} {d.tif}  | {d.rationale}")

    # 3) Fuzzy-collapse ablation ------------------------------------------------
    hr("3. Fuzzy-collapse ablation -- WITHDRAWN")
    print("  This demo no longer prints a fuzzy-collapse delta.")
    print()
    print("  The ablation was withdrawn on review: it is arithmetic, not an")
    print("  experiment. Collapsing three classes to two always loses the")
    print("  minority class, so any I/F-merged model is capped at 0.665 by the")
    print("  label prior alone, and the reported collapsed figure is recoverable")
    print("  from the published confusion matrix by summing two columns.")
    print()
    print("  eval.fuzzy_collapse_ablation() is still importable and unchanged;")
    print("  it is simply no longer presented as evidence. See RESULTS.md,")
    print("  section RETRACTED, and memo finding F4.")


if __name__ == "__main__":
    main()
