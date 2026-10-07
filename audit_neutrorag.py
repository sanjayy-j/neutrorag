"""
audit_neutrorag.py -- structural audit of the NeutroRAG scorer.

Drop this in the repo root (next to demo.py) and run:  python audit_neutrorag.py

Every check below is deterministic and needs no network, no model download and
no FEVER access. Each one probes a claim the report makes about the method.
"""
import random

from neutrorag.core import TIF
from neutrorag.fever import _examples_from_rows, load_sample
from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer

LAM = 0.5
R_HARDCODED = 0.9  # neutrorag/fever.py sets Evidence(..., relevance=0.9) everywhere


def hr(n, title):
    print("\n" + "=" * 74)
    print(f"CHECK {n}. {title}")
    print("=" * 74)


# --------------------------------------------------------------------------
hr(1, "Is T+I+F actually free?  (report: 'independent memberships')")
# T+I+F = r(e+n+c) + (1-r) + lam*disagree = 1 + lam*disagree, since e+n+c = 1.
random.seed(0)
viol = 0
for _ in range(200_000):
    e, n, c = (random.random() for _ in range(3))
    s = e + n + c
    e, n, c = e / s, n / s, c / s
    r = random.random()
    T, F, I = r * e, r * c + LAM * 2 * min(e, c), 1 - r + r * n
    if min(T, I, F) < 0 or max(T, I, F) > 1:
        continue
    if abs((T + I + F) - (1 + LAM * 2 * min(e, c))) > 1e-9:
        viol += 1
print(f"  counterexamples to  T+I+F == 1 + lambda*disagree  in 200k draws: {viol}")
print("  => the triple has ONE degree of freedom beyond the probability simplex,")
print("     not three. 'Mass exceeds 1' is a restatement of 'disagree > 0'.")


# --------------------------------------------------------------------------
hr(2, "What does the transform do to the argmax?  (r = 0.9, hardcoded)")
r = R_HARDCODED
print(f"  pick T:  raw NLI  e > n          |  NeutroRAG  e > n + {(1-r)/r:.3f}")
print(f"  pick F:  raw NLI  c > e          |  NeutroRAG  c > {r/(r+1):.3f}*e        (when c < e)")
print(f"  pick F:  raw NLI  c > n          |  NeutroRAG  c > {(1-r)/(r+1):.4f} + {r/(r+1):.4f}*n")
print("  => T must clear an extra margin; F wins on roughly half the contradiction")
print("     mass it would normally need; I gets a free +0.10 floor from (1-r).")
print("     This is the mechanism behind SUPPORTS->I being the top error cell.")


# --------------------------------------------------------------------------
hr(3, "The missing baseline: argmax(T,I,F) vs argmax(e,n,c)")
nli = MockNLIBackend()
scorer = NLIRetrievalScorer(nli=nli)
CH = {"SUPPORTS": "T", "NOT ENOUGH INFO": "I", "REFUTES": "F"}
neu_ok = raw_ok = 0
for ex in load_sample():
    tif = scorer.score(ex.claim, ex.evidence)
    ws = [e.relevance for e in ex.evidence]
    ps = [nli.predict(e.text, ex.claim) for e in ex.evidence]
    W = sum(ws)
    e_ = sum(w * p[0] for w, p in zip(ws, ps)) / W
    n_ = sum(w * p[1] for w, p in zip(ws, ps)) / W
    c_ = sum(w * p[2] for w, p in zip(ws, ps)) / W
    raw = max((("T", e_), ("I", n_), ("F", c_)), key=lambda kv: kv[1])[0]
    g = CH[ex.gold_label]
    neu_ok += tif.dominant == g
    raw_ok += raw == g
print(f"  (mock backend, 12 embedded examples -- illustrative only)")
print(f"  neutrosophic argmax : {neu_ok}/12")
print(f"  raw NLI argmax      : {raw_ok}/12")
print("  => ~15 lines to compute. Run this on the real 200-claim FEVER pass")
print("     BEFORE anything else. If raw NLI also lands near 0.745, the")
print("     neutrosophic layer contributed nothing to the headline number.")


# --------------------------------------------------------------------------
hr(4, "Does the mock backend emit a valid distribution?")
pairs = [
    ("Sharks are not mammals; sharks are a group of fish.", "Sharks are mammals."),
    ("The heart is a muscular organ that pumps blood.", "The heart pumps blood."),
]
for prem, hyp in pairs:
    e, n, c = nli.predict(prem, hyp)
    print(f"  e={e:.2f} n={n:.2f} c={c:.2f}   SUM = {e+n+c:.3f}")
print("  => e+n+c = 1.1, not 1. clamp() in MockNLIBackend.predict truncates the")
print("     neutral mass at 0 instead of renormalising. Every demo.py number and")
print("     every mock-driven unit test runs on a non-distribution, which is also")
print("     why the README prints F=1.00 (analytically F <= 0.95 on real inputs).")


# --------------------------------------------------------------------------
hr(5, "Empty-evidence shortcut on gold-I rows")
rows = [
    {"claim": "Some unverifiable claim.", "label": "NOT ENOUGH INFO", "evidence": []},
    {"claim": "Another one.", "label": "NOT ENOUGH INFO", "evidence": [["Page", 3, ""]]},
]
for ex in _examples_from_rows(rows):
    tif = scorer.score(ex.claim, ex.evidence)
    print(f"  evidence kept={len(ex.evidence)} -> {tif}  pred={tif.dominant}  gold={ex.gold_channel}")
print("  => _examples_from_rows drops malformed/empty evidence silently; the")
print("     scorer then returns TIF(0,1,0) and scores a guaranteed-correct I.")
print("     COUNT how many of the 200 FEVER rows hit this path. Those are free")
print("     points on exactly the class the paper's argument depends on.")


# --------------------------------------------------------------------------
hr(6, "Ceiling on the fuzzy-collapse ablation")
T_n, I_n, F_n, N = 65, 67, 68, 200
ceil = (T_n + max(I_n, F_n)) / N
print(f"  gold counts from the reported confusion matrix: T={T_n} I={I_n} F={F_n}")
print(f"  any I/F-merged model, even with a PERFECT T detector, tops out at")
print(f"      ({T_n} + {max(I_n,F_n)}) / {N} = {ceil:.3f}")
print(f"  reported: collapsed 0.540, full 0.745")
print("  => the delta is bounded by how many gold-I items exist, not by anything")
print("     about representational power. Collapsing 3 classes to 2 always loses")
print("     the minority class. This is arithmetic, not an experiment.")


# --------------------------------------------------------------------------
hr(7, "Does conflict survive relevance-weighted mean pooling?")
claim = "Drug X reduces the risk of Condition Y in high-risk patients."
support = Evidence("Drug X reduced the risk of Condition Y by 30 percent in high-risk patients.", 1.0)
refute = Evidence("A large meta-analysis found no significant reduction in Condition Y "
                  "risk among high-risk patients taking Drug X.", 1.0)
filler = [
    Evidence("Drug X is commonly prescribed to high-risk patients for a range of conditions.", 1.0),
    Evidence("Drug X was approved for cardiovascular risk reduction over a decade ago.", 0.84),
    Evidence("Condition Y prevalence increases significantly with age.", 0.74),
    Evidence("The stock market fell sharply on Tuesday amid inflation concerns.", 0.53),
]
for k in range(len(filler) + 1):
    tif = scorer.score(claim, [support, refute] + filler[:k])
    print(f"  2 conflicting + {k} neutral -> {tif}")
print("  => F decays monotonically as irrelevant passages arrive. The conflict")
print("     between the two passages has not changed; only the pooling has.")
print("     Suggested fix: disagree = max over PAIRS of min(e_i, c_j), not a")
print("     function of pooled means. Conflict is a set-level property.")


# --------------------------------------------------------------------------
hr(8, "Has the expansion loop ever been scored by the real scorer?")
calls = []


def real_score(claim, evidence):
    tif = scorer.score(claim, evidence)
    calls.append(tif)
    return tif


def expand_noop(claim, evidence, step):
    return evidence  # what a retriever returns when it has nothing new


from neutrorag.policy import NeutroPolicy  # noqa: E402

d = NeutroPolicy().run("The bridge was painted blue in 2005.",
                       [Evidence("The bridge spans a wide river in the northern region.", 0.3)],
                       real_score, expand_noop)
print(f"  triples seen across the loop: {[repr(t) for t in calls]}")
print(f"  final action: {d.action}  after {d.expansions_used} expansions")
print("  => tests/test_policy.py drives run() with hand-written score_fn stubs")
print("     that return hardcoded triples. The loop has never been exercised")
print("     against NLIRetrievalScorer.score, which is why it is a no-op in the")
print("     pipeline screenshot (identical TIF at steps 0, 1 and 2).")
