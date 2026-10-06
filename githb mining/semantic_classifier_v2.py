"""
semantic_classifier_v2.py  —  HYBRID classifier

V1 (LLM does everything) gold audit showed the LLM cannot reliably assign
`architecture_primary`: prompt iteration fixed 0/3 target errors and even
regressed GC07, because a 7B model anchors on surface cues and invents
`sequential` for systems with no structural evidence. A deterministic precedence
rule over the extracted structural evidence scores 9/9 on the gold architectures.

V2 is therefore hybrid:
    architecture_primary  <- deterministic rule (architecture_rule.py)   [reproducible]
    specialization        <- LLM, from responsibilities                  [semantic]
    confidence            <- LLM

Specialization/confidence are REUSED from V1 by default (data/semantic_classifications_v1.json).
Rationale: on the gold set V1's specialization matched 6/8 vs 4/8 for the
responsibilities-only prompt draft, and reuse avoids a fragile full re-run.
Use --reclassify-spec to instead re-run specialization with the responsibilities-only
prompt below (writes to the same v2 file).

Modes:
    python semantic_classifier_v2.py --dryrun      # rule arch distribution over 333 (no LLM)
    python semantic_classifier_v2.py --build       # write data/semantic_classifications_v2.json (default)
    python semantic_classifier_v2.py --reclassify-spec --ollama   # optional: re-run spec via LLM

V1 outputs are preserved as the baseline.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import semantic_classifier as v1
from architecture_rule import architecture_primary

DATA_DIR = Path("data")
V1_RESULT = DATA_DIR / "semantic_classifications_v1.json"
RESULT_OUT = DATA_DIR / "semantic_classifications_v2.json"

# Responsibilities-only specialization prompt (kept for --reclassify-spec). This
# forbids using any architectural property as evidence of specialization.
SPEC_SYSTEM_PROMPT = """You are a research data annotator for a study of open-source LLM-based multi-agent systems.

Your ONLY job is to rate responsibility specialization from the supplied source-code evidence.

Judge specialization EXCLUSIVELY by the FUNCTIONAL distinctness of the agents'
responsibilities. Do NOT use execution order, hierarchy, parallelism, routing, or any
other architectural property as evidence. Do NOT rate by the NUMBER of agents. Two agents
in a sequential pipeline can be highly specialized; many agents under a supervisor can be
low-specialization.

Decisive test — do the agents perform DIFFERENT FUNCTIONS, or the SAME function on
different inputs/parameters?
- low: agents share essentially the same role/function, or differ only trivially/generically.
- medium: agents differ but with meaningful overlap, OR perform the SAME function differing
  only by a parameter such as language, region, topic, or data source
  (e.g. an English assistant and a Spanish assistant = medium, not high).
- high: each agent owns a functionally DISTINCT, complementary responsibility
  (e.g. one researches, another writes, another reviews = high, even if only two agents).
- unresolved: insufficient responsibility evidence.

Return ONLY valid JSON: {"specialization":"low|medium|high|unresolved","confidence":"high|medium|low","rationale":"brief"}
"""


def atomic_write(path, obj):
    DATA_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_v1_spec():
    """system_id -> (specialization, confidence, rationale) from V1."""
    out = {}
    if V1_RESULT.exists():
        for r in json.loads(V1_RESULT.read_text(encoding="utf-8")):
            c = r.get("classification", {})
            out[r["system_id"]] = (
                c.get("specialization", "unresolved"),
                c.get("confidence", "low"),
                c.get("rationale", ""),
            )
    return out


def dryrun():
    bundles = v1.load_bundles()
    dist = Counter(architecture_primary(b) for b in bundles)
    n = len(bundles)
    print(f"RULE-BASED architecture_primary distribution (n={n}):")
    for k, v in dist.most_common():
        print(f"  {k:20s} {v:4d}  {100*v/n:5.1f}%")
    resolved = n - dist.get("unresolved", 0)
    print(f"\nResolved (evidence-grounded) systems: {resolved}/{n} ({100*resolved/n:.1f}%)")


def build(model=None, host=None, reclassify_spec=False):
    bundles = v1.load_bundles()
    v1spec = load_v1_spec()
    results = []
    for b in bundles:
        sid = b["system_id"]
        arch = architecture_primary(b)
        if reclassify_spec:
            try:
                r = v1.call_ollama(model, SPEC_SYSTEM_PROMPT + "\n\nEVIDENCE:\n"
                                   + json.dumps(b, ensure_ascii=False, indent=2), host)
                spec = r.get("specialization", "unresolved")
                conf = r.get("confidence", "low")
                rat = r.get("rationale", "")
                if spec not in v1.ALLOWED_SPECIALIZATION:
                    spec, conf, rat = "unresolved", "low", "invalid specialization"
            except Exception as exc:
                spec, conf, rat = "unresolved", "low", f"spec error: {type(exc).__name__}: {exc}"
        else:
            spec, conf, rat = v1spec.get(sid, ("unresolved", "low", "no v1 label"))
        results.append({
            "system_id": sid,
            "repository": b["repository"],
            "classification": {
                "specialization": spec,
                "architecture_primary": arch,
                "architecture_method": "rule",          # provenance: not LLM
                "specialization_method": "llm_v2" if reclassify_spec else "llm_v1_reused",
                "confidence": conf,
                "rationale": rat,
                "architecture_patterns": b.get("architecture_patterns", []),
            },
        })
        if reclassify_spec:
            atomic_write(RESULT_OUT, results)  # checkpoint the slow LLM path
    atomic_write(RESULT_OUT, results)
    print(f"Saved {len(results)} systems -> {RESULT_OUT}")
    dist = Counter(r["classification"]["architecture_primary"] for r in results)
    print("architecture_primary:", dict(dist.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true", help="Rule arch distribution only (no write, no LLM).")
    ap.add_argument("--build", action="store_true", help="Write the hybrid v2 dataset (default).")
    ap.add_argument("--reclassify-spec", action="store_true", help="Re-run specialization via LLM instead of reusing V1.")
    ap.add_argument("--ollama", action="store_true", help="(with --reclassify-spec) enable LLM calls.")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    if args.dryrun:
        dryrun()
        return
    build(model=args.model, host=args.host,
          reclassify_spec=(args.reclassify_spec and args.ollama))


if __name__ == "__main__":
    main()
