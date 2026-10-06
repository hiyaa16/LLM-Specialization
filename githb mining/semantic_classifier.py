import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
EVIDENCE = DATA_DIR / "evidence_v6_6.json"
INPUT_OUT = DATA_DIR / "classification_input_v1.json"
RESULT_OUT = DATA_DIR / "semantic_classifications_v1.json"

ALLOWED_ARCHITECTURES = [
    "single_agent",
    "sequential",
    "parallel",
    "hierarchical",
    "peer_to_peer",
    "conditional_routing",
    "iterative_feedback",
    "event_driven",
    "graph_conditional",
    "hybrid",
    "unresolved",
]

ALLOWED_SPECIALIZATION = ["low", "medium", "high", "unresolved"]

SYSTEM_PROMPT = """You are a research data annotator for a study of open-source LLM-based multi-agent systems.

Your job is semantic classification ONLY. The input is evidence extracted from source code.

Golden rule:
EXTRACTED EVIDENCE IS THE SOURCE OF TRUTH. Do not invent agents, responsibilities, edges, or architecture mechanisms that are not supported by the evidence.

Classify:
1. responsibility specialization: low / medium / high / unresolved
2. primary architecture: one allowed architecture label
3. confidence: high / medium / low
4. short rationale grounded only in the supplied evidence.

SPECIALIZATION:
- low: agents have largely overlapping/general responsibilities or differentiation is weak.
- medium: agents have some distinct responsibilities but meaningful overlap remains.
- high: agents have clearly differentiated, complementary responsibilities.
- unresolved: insufficient responsibility evidence.

ARCHITECTURE:
Use the strongest structural evidence. Do NOT infer an architecture merely because a framework has a default behavior.
- sequential: explicit ordered/dependent execution
- parallel: explicit fan-out/concurrent workers or broadcast execution
- hierarchical: explicit supervisor/delegation/manager structure
- peer_to_peer: direct agent-to-agent communication without a central coordinator
- conditional_routing: explicit conditional routing/handoff/selection
- iterative_feedback: explicit critique/revision/repeated feedback loop
- event_driven: explicit publish/subscribe or producer-consumer event mechanism
- graph_conditional: explicit graph with conditional routing when that graph structure is the primary organization
- hybrid: two or more major architectural mechanisms are jointly essential and no single one clearly dominates
- single_agent: only one application-level agent in the system
- unresolved: evidence is insufficient.

IMPORTANT:
- A Team/Crew/Workflow/orchestration object is not automatically an agent.
- Membership alone does not prove sequential execution.
- Interaction primitives such as composition, round_robin_group, nested_team_composition, etc. are evidence; interpret them conservatively.
- Preserve ambiguity when the evidence does not support a confident label.

Return ONLY valid JSON with this schema:
{
  "specialization": "low|medium|high|unresolved",
  "architecture_primary": "single_agent|sequential|parallel|hierarchical|peer_to_peer|conditional_routing|iterative_feedback|event_driven|graph_conditional|hybrid|unresolved",
  "confidence": "high|medium|low",
  "rationale": "brief evidence-grounded explanation"
}
"""

def clean_text(x, limit=1800):
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x if len(x) <= limit else x[:limit] + " ..."

def build_bundle(system):
    agents = []
    for a in system.get("agents", []):
        responsibilities = []
        for r in a.get("responsibilities", []):
            responsibilities.append({
                "field": r.get("field"),
                "text": clean_text(r.get("text"), 1200),
                "line": r.get("line"),
                "evidence_type": r.get("evidence_type"),
            })
        agents.append({
            "symbol": a.get("symbol"),
            "name": a.get("name"),
            "file": a.get("file"),
            "line": a.get("line"),
            "responsibilities": responsibilities,
            "confidence": a.get("confidence"),
            "provenance": a.get("provenance"),
        })

    interactions = []
    for e in system.get("interactions", []):
        interactions.append({
            "source_agent": e.get("source_agent"),
            "target_agent": e.get("target_agent"),
            "type": e.get("type"),
            "evidence_type": e.get("evidence_type"),
            "file": e.get("file"),
            "line": e.get("line"),
            "evidence": e.get("evidence"),
        })

    return {
        "system_id": system.get("system_id"),
        "repository": system.get("repository"),
        "files": system.get("files", []),
        "anchor_files": system.get("anchor_files", []),
        "provenance": system.get("provenance", []),
        "system_boundary_status": system.get("system_boundary_status"),
        "agent_count": system.get("agent_count"),
        "agents": agents,
        "interaction_evidence": interactions,
        "architecture_patterns": system.get("architecture_patterns", []),
        "interaction_primitives": system.get("interaction_primitives", []),
        "architecture_evidence_types": system.get("architecture_evidence_types", []),
        "composition_evidence": system.get("composition_evidence", []),
        "explicit_interaction_edges": system.get("explicit_interaction_edges", []),
        "confidence": system.get("confidence"),
    }

def load_bundles():
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    bundles = []
    for repo in data:
        for system in repo.get("systems", []):
            bundles.append(build_bundle(system))
    return bundles

def make_prompt(bundle):
    return SYSTEM_PROMPT + "\n\nSOURCE-CODE EVIDENCE BUNDLE:\n" + json.dumps(
        bundle, ensure_ascii=False, indent=2
    )

def prepare():
    DATA_DIR.mkdir(exist_ok=True)
    bundles = load_bundles()
    prompts = [
        {
            "system_id": b["system_id"],
            "repository": b["repository"],
            "prompt": make_prompt(b),
        }
        for b in bundles
    ]
    INPUT_OUT.write_text(
        json.dumps(bundles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DATA_DIR / "classification_prompts_v1.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(bundles)} candidate systems.")
    print(f"Evidence bundles: {INPUT_OUT}")
    print(f"LLM prompts:      {DATA_DIR / 'classification_prompts_v1.json'}")

def call_ollama(model, prompt, host):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return json.loads(raw["response"])

def classify_ollama(model, host):
    bundles = load_bundles()

    # Resume support: reload any results already written on a previous run and
    # skip those systems. Entries that previously errored out are NOT treated
    # as done, so they get re-attempted (a killed run often leaves transient
    # failures worth retrying).
    done = {}
    if RESULT_OUT.exists():
        try:
            for r in json.loads(RESULT_OUT.read_text(encoding="utf-8")):
                rationale = r.get("classification", {}).get("rationale", "")
                if not str(rationale).startswith("Classifier error"):
                    done[r["system_id"]] = r
        except Exception as exc:
            print(f"[WARN] Could not read existing results ({exc}); starting fresh.", flush=True)
            done = {}

    results = list(done.values())

    def flush():
        # Atomic checkpoint: write to a temp file in the same directory, then
        # os.replace() it over the target. A rename on one filesystem is
        # atomic, so an abrupt kill can never leave a half-written (corrupt)
        # results file — the target is always either the previous complete
        # version or the new complete version.
        DATA_DIR.mkdir(exist_ok=True)
        tmp = RESULT_OUT.with_suffix(RESULT_OUT.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, RESULT_OUT)

    total = len(bundles)
    if done:
        print(f"Resuming: {len(done)}/{total} already classified.", flush=True)

    for i, bundle in enumerate(bundles, 1):
        sid = bundle["system_id"]
        if sid in done:
            print(f"[{i}/{total}] {sid} (skip: already done)", flush=True)
            continue
        print(f"[{i}/{total}] {sid}", flush=True)
        try:
            result = call_ollama(model, make_prompt(bundle), host)
            if result.get("specialization") not in ALLOWED_SPECIALIZATION:
                raise ValueError("invalid specialization")
            if result.get("architecture_primary") not in ALLOWED_ARCHITECTURES:
                raise ValueError("invalid architecture_primary")
            if result.get("confidence") not in {"high", "medium", "low"}:
                raise ValueError("invalid confidence")
            entry = {
                "system_id": sid,
                "repository": bundle["repository"],
                "classification": result,
            }
        except Exception as exc:
            entry = {
                "system_id": sid,
                "repository": bundle["repository"],
                "classification": {
                    "specialization": "unresolved",
                    "architecture_primary": "unresolved",
                    "confidence": "low",
                    "rationale": f"Classifier error; retained for review: {type(exc).__name__}: {exc}",
                },
            }
        results.append(entry)
        # Checkpoint after every system so an interruption never loses more
        # than the one in flight.
        flush()

    print(f"\nSaved: {RESULT_OUT}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true",
                        help="Create evidence bundles and prompts without calling an LLM.")
    parser.add_argument("--ollama", action="store_true",
                        help="Run classification through a local Ollama server.")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Local Ollama model name.")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama base URL.")
    args = parser.parse_args()

    if not args.prepare and not args.ollama:
        args.prepare = True

    if args.prepare:
        prepare()
    if args.ollama:
        classify_ollama(args.model, args.host)

if __name__ == "__main__":
    main()
