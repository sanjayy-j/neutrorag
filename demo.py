"""NeutroRAG demo -- SCAFFOLD ONLY. Every number it prints is illustrative.

    python demo.py

Shows three things:
  1. the FEVER -> (T, I, F) channel check on 12 hand-written examples,
  2. the five-way decision policy on illustrative cases,
  3. a note recording that the fuzzy-collapse ablation is WITHDRAWN.

NOTHING HERE IS A RESULT. The backend is MockNLIBackend, a lexical-overlap
heuristic and not an NLI model, and the data is 12 toy examples embedded in
fever.py. Quoting any figure from this script as a finding would be an error.

The project's measured numbers, the tests that produced them and the figures
that are retracted are all in RESULTS.md, with results/claims.md as the
authoritative manifest. Among those: the headline FEVER attribution is
RETRACTED (a raw argmax(e,n,c) baseline on identical model calls scores
higher), and the fuzzy-collapse ablation is WITHDRAWN as arithmetic rather
than experiment -- so there is no "paper-grade" version of this script's
output to swap a real backend in for. See RESULTS.md section 3 for what the
real evaluation found instead.
"""
from neutrorag import Evidence, NeutroPolicy, NLIRetrievalScorer, MockNLIBackend
# fuzzy_collapse_ablation is deliberately NOT imported: this script no longer
# computes it. It remains available in neutrorag.eval for anyone reproducing
# the retracted figure, but importing it here would imply it is still used.
from neutrorag.eval import failure_type_attribution, format_confusion
from neutrorag.fever import load_sample, score_examples


def hr(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    policy = NeutroPolicy()

    banner = "  MOCK BACKEND -- SCAFFOLD OUTPUT -- NOT A RESULT  "
    print("#" * 68)
    print("#" + banner.center(66) + "#")
    print("#" * 68)
    print("NeutroRAG demo. Every number below is illustrative.")
    print()
    print("  Backend : MockNLIBackend -- a lexical-overlap heuristic.")
    print("            It is NOT an NLI model and does no inference.")
    print("  Data    : 12 hand-written examples embedded in fever.py.")
    print("  Status  : shows that the pipeline runs. Nothing here is a")
    print("            finding, and no figure here may be quoted as one.")
    print()
    print("  Measured results, and the three retracted figures, are in")
    print("  RESULTS.md; results/claims.md is the authoritative manifest.")
    print("#" * 68)

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

    print("  Every row below is mock-backend output on a hand-built case.\n")
    for name, ev in cases.items():
        d = policy.run(claims[name], ev, scorer.score, expand_fn)
        print(f"  [mock] {name:22s} -> {d.action.value:22s} {d.tif}  "
              f"| {d.rationale}")

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
    print("  eval.fuzzy_collapse_ablation() remains in neutrorag.eval,")
    print("  unchanged, so the retracted figure stays reproducible. It is")
    print("  simply no longer computed or presented here. See RESULTS.md,")
    print("  section RETRACTED, and memo finding F4.")

    print("\n" + "#" * 68)
    print("#" + "  END -- every figure above was MOCK-BACKEND SCAFFOLD  "
          .center(66) + "#")
    print("#" + "  Real numbers: RESULTS.md / results/claims.md  "
          .center(66) + "#")
    print("#" * 68)


if __name__ == "__main__":
    main()
