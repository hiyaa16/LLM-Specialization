"""
V6.6-compatible gold-case validator.

Purpose:
- Validate the EVIDENCE produced by code_miner_v6_6.py.
- Do NOT re-run the miner.
- Do NOT import code_miner_v3_2.py.
- Keep extractor failures separate from semantic-normalization issues.

Usage (from the project root):
    python validate.py

Expected input:
    data/evidence_v6_6.json
    gold_cases.json

Output:
    data/validation_v6_6.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parent
EVIDENCE_FILE = ROOT / "data" / "evidence_v6_6.json"
GOLD_FILE = ROOT / "gold_cases.json"
OUTPUT_FILE = ROOT / "data" / "validation_v6_6.txt"

# Semantic layer produced by semantic_classifier.py. The V6.6 miner is
# deliberately evidence-preserving and leaves architecture_primary /
# specialization null on every system; those labels are assigned downstream
# by the semantic classifier. To validate the classifier's judgments against
# the gold cases (not just the miner's structural extraction), we merge these
# labels back onto the matching systems by system_id before running checks.
CLASSIFICATIONS_FILE = ROOT / "data" / "semantic_classifications_v1.json"


# ---------------------------------------------------------------------
# Gold source locations
# These are deliberately exact where the gold case has a known source.
# ---------------------------------------------------------------------

EXACT_FILES = {
    "GC01": ["examples/handoffs/message_filter.py"],
    "GC02": ["contributing/samples/multi_agent/sub_agents/agent.py"],
    "GC03": ["contributing/samples/workflows/parallel_worker/agent.py"],
    "GC04": ["contributing/samples/legacy_workflows/non_llm_sequential/agent.py"],
    "GC05": ["python/samples/agentchat_chainlit/app_team.py"],
    "GC06": ["agents.py", "main.py"],
    "GC07": ["agents/graph.py"],
    "GC08": ["examples/build_customized_multi_agents.py"],
    "GC09": ["cookbook/03_teams/01_quickstart/nested_teams.py"],
    "GC10": ["src/agentscope/agent/_agent.py"],
}


# Known repository names for the 10 gold cases.
GOLD_REPOS = {
    "GC01": ["openai/openai-agents-python"],
    "GC02": ["google/adk-python"],
    "GC03": ["google/adk-python"],
    "GC04": ["google/adk-python"],
    "GC05": ["microsoft/autogen"],
    "GC06": ["bhancockio/crewai-updated-tutorial-hierarchical"],
    "GC07": ["ki11e6/langgraph-multi-agent"],
    "GC08": ["FoundationAgents/MetaGPT"],
    "GC09": ["agno-agi/agno"],
    "GC10": ["agentscope-ai/agentscope"],
}


# Gold expectations.  The validator uses these only as diagnostics;
# gold_cases.json remains the source of truth if it contains equivalent
# information.
EXPECTED = {
    "GC01": {
        "is_system": True,
        "agents": 3,
        "specialization": "medium",
        "architecture": "conditional_routing",
        "patterns": {"conditional_handoff", "conditional_routing"},
        "min_edges": 1,
    },
    "GC02": {
        "is_system": True,
        "agents": 3,
        "specialization": "high",
        "architecture": "hierarchical",
        "patterns": {"hierarchical_delegation", "hierarchical"},
        "min_edges": 2,
    },
    "GC03": {
        "is_system": True,
        "agents": 2,
        "specialization": "high",
        "architecture": "parallel",
        "patterns": {"fan_out_fan_in", "parallel"},
        "min_edges": 0,
    },
    "GC04": {
        "is_system": True,
        "agents": 2,
        "specialization": "low",
        "architecture": "sequential",
        "patterns": {"sequential_dependency", "sequential"},
        "min_edges": 1,
    },
    "GC05": {
        "is_system": True,
        "agents": 2,
        "specialization": "high",
        "architecture": "iterative_feedback",
        "patterns": {"round_robin_group", "iterative_feedback"},
        "min_edges": 0,
    },
    "GC06": {
        "is_system": True,
        "agents": 4,
        "specialization": "high",
        "architecture": "hierarchical",
        "patterns": {"hierarchical_delegation", "sequential_task_dependency"},
        "min_edges": 0,
    },
    "GC07": {
        "is_system": True,
        "agents": 4,
        "specialization": "high",
        # The V6.6 miner intentionally does not have to assign a single
        # semantic primary architecture here. Treat this as REVIEW rather
        # than an extractor failure if structural evidence is present.
        "architecture": "conditional_graph",
        "patterns": {
            "conditional_routing",
            "sequential_dependency",
            "iterative_feedback",
            "human_in_the_loop",
        },
        "min_edges": 3,
    },
    "GC08": {
        "is_system": True,
        "agents": 3,
        "specialization": "high",
        "architecture": "event_driven",
        "patterns": {"publish_subscribe", "sequential_dependency", "feedback_loop"},
        "min_edges": 0,
    },
    "GC09": {
        "is_system": True,
        "agents": 4,
        # V6.6 intentionally represents nested team structure as evidence,
        # not necessarily as a final semantic "hierarchical" label.
        "architecture": "hierarchical_nested_team",
        "patterns": {
            "nested_team_composition",
            "hierarchical_coordination",
            "sequential_team_dependency",
        },
        "min_edges": 0,
    },
    "GC10": {
        "is_system": False,
        "agents": 0,
        "specialization": "na",
        "architecture": "na",
        "patterns": set(),
        "min_edges": 0,
    },
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def norm_path(p: Any) -> str:
    if p is None:
        return ""
    s = str(p).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def path_matches(actual: str, expected: str) -> bool:
    """Exact suffix-aware source matching."""
    a = norm_path(actual)
    e = norm_path(expected)
    if not a or not e:
        return False
    return a == e or a.endswith("/" + e)


def get_repo_name(repo: Dict[str, Any]) -> str:
    for key in ("full_name", "repo", "repository", "name", "fullName"):
        value = repo.get(key)
        if isinstance(value, str):
            return value
    return ""


def get_systems(repo: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("systems", "system_detections", "detected_systems"):
        value = repo.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def get_agents(system: Dict[str, Any]) -> List[Any]:
    for key in ("agents", "agent_instances", "members", "agent_nodes"):
        value = system.get(key)
        if isinstance(value, list):
            return value
    return []


def get_edges(system: Dict[str, Any]) -> List[Any]:
    for key in ("edges", "interactions", "interaction_edges", "structural_edges"):
        value = system.get(key)
        if isinstance(value, list):
            return value
    return []


def get_patterns(system: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()

    for key in ("architecture_patterns", "patterns", "interaction_primitives"):
        value = system.get(key)
        if isinstance(value, list):
            out.update(str(x).strip().lower() for x in value if x is not None)
        elif isinstance(value, str):
            out.add(value.strip().lower())

    # Evidence records can carry semantic types.
    evidence = system.get("structural_evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                for key in ("type", "evidence_type", "interaction_type"):
                    value = item.get(key)
                    if isinstance(value, str):
                        out.add(value.strip().lower())

    return out


def get_architecture(system: Dict[str, Any]) -> Optional[str]:
    for key in ("architecture_primary", "architecture", "architecture_type"):
        value = system.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def get_specialization(system: Dict[str, Any]) -> Optional[str]:
    for key in (
        "specialization",
        "specialization_level",
        "specialization_label",
        "responsibility_specialization",
    ):
        value = system.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def source_files(system: Dict[str, Any]) -> Set[str]:
    found: Set[str] = set()

    # Common source-file fields.
    for key in (
        "source_file",
        "file",
        "path",
        "anchor_file",
        "source",
        "files",
        "source_files",
        "anchor_files",
    ):
        value = system.get(key)
        if isinstance(value, str):
            found.add(norm_path(value))
        elif isinstance(value, list):
            for x in value:
                if isinstance(x, str):
                    found.add(norm_path(x))
                elif isinstance(x, dict):
                    for kk in ("path", "file", "source_file", "source"):
                        if isinstance(x.get(kk), str):
                            found.add(norm_path(x[kk]))

    # Search one level inside evidence objects.
    for key in ("structural_evidence", "evidence", "architecture_evidence"):
        value = system.get(key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                for kk in ("file", "source_file", "path", "anchor_file"):
                    if isinstance(item.get(kk), str):
                        found.add(norm_path(item[kk]))

    return {x for x in found if x}


def system_matches_files(system: Dict[str, Any], expected_files: List[str]) -> bool:
    actual = source_files(system)
    if not actual:
        return False

    return all(
        any(path_matches(a, e) for a in actual)
        for e in expected_files
    )


def repo_matches(repo_name: str, expected_names: List[str]) -> bool:
    r = repo_name.lower().strip()
    return any(r == x.lower() for x in expected_names)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_repo_records(data: Any) -> List[Dict[str, Any]]:
    """
    Handles common V6.6 shapes:
      [{"full_name": ..., "systems": [...]}, ...]
      {"repositories": [...]}
      {"results": [...]}
      {"repos": [...]}
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("repositories", "results", "repos", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        # Single repo record.
        if any(k in data for k in ("full_name", "systems", "source_files")):
            return [data]

    return []


def merge_semantic_classifications(repos: List[Dict[str, Any]]) -> int:
    """Inject architecture_primary / specialization from the semantic
    classifier onto each system, joined by system_id. Returns the number of
    systems that received a label. No-op (returns 0) if the classifications
    file is absent, so the validator still runs on raw evidence alone."""
    if not CLASSIFICATIONS_FILE.exists():
        return 0

    try:
        raw = load_json(CLASSIFICATIONS_FILE)
    except Exception:
        return 0

    by_id: Dict[str, Dict[str, Any]] = {}
    for row in raw if isinstance(raw, list) else []:
        sid = row.get("system_id")
        cls = row.get("classification")
        if isinstance(sid, str) and isinstance(cls, dict):
            by_id[sid] = cls

    merged = 0
    for repo in repos:
        for system in get_systems(repo):
            sid = system.get("system_id")
            cls = by_id.get(sid)
            if not cls:
                continue
            arch = cls.get("architecture_primary")
            spec = cls.get("specialization")
            # Only overwrite when the classifier produced a concrete label;
            # leave the miner's null in place otherwise so "unresolved" is
            # still visibly distinct from "not classified".
            if isinstance(arch, str) and arch:
                system["architecture_primary"] = arch
            if isinstance(spec, str) and spec:
                system["specialization"] = spec
            system["classifier_confidence"] = cls.get("confidence")
            merged += 1
    return merged


def load_gold() -> Dict[str, Dict[str, Any]]:
    """
    Prefer gold_cases.json, but retain the embedded expectations above so
    the validator remains usable if the older gold file has a slightly
    different schema.
    """
    gold = {k: dict(v) for k, v in EXPECTED.items()}

    if not GOLD_FILE.exists():
        return gold

    try:
        raw = load_json(GOLD_FILE)
    except Exception:
        return gold

    # Accept either a list of cases or {"cases": [...]}.
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        return gold

    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = case.get("id") or case.get("case_id") or case.get("name")
        if not isinstance(cid, str) or cid not in gold:
            continue

        # Accept both schemas:
        #   {"specialization": "...", ...}
        # and
        #   {"expected": {"specialization": "...", ...}}
        # without changing the embedded exact-source mapping.
        expected_block = case.get("expected")
        if not isinstance(expected_block, dict):
            expected_block = {}

        agent_values = (
            case.get("expected_agents"),
            case.get("expected_agent_count"),
            expected_block.get("agents"),
            expected_block.get("agent_instances"),
            expected_block.get("expected_agents"),
            expected_block.get("expected_agent_count"),
        )
        for value in agent_values:
            if isinstance(value, int):
                gold[cid]["agents"] = value
                break

        for key in ("specialization", "architecture"):
            value = case.get(key)
            if not isinstance(value, str):
                value = expected_block.get(key)
            if isinstance(value, str):
                gold[cid][key] = value.lower()

        pattern_values = (
            case.get("expected_patterns"),
            case.get("interaction_patterns"),
            case.get("patterns"),
            expected_block.get("expected_patterns"),
            expected_block.get("interaction_patterns"),
            expected_block.get("patterns"),
        )
        for value in pattern_values:
            if isinstance(value, list):
                gold[cid]["patterns"] = {str(x).lower() for x in value}
                break

        if isinstance(case.get("is_system"), bool):
            gold[cid]["is_system"] = case["is_system"]

    return gold


# ---------------------------------------------------------------------
# Pattern equivalence
# ---------------------------------------------------------------------

PATTERN_ALIASES = {
    "conditional_handoff": {
        "conditional_handoff",
        "explicit_handoff",
        "handoff",
        "conditional_routing",
    },
    "hierarchical_delegation": {
        "hierarchical_delegation",
        "explicit_sub_agent_delegation",
        "sub_agent_delegation",
        "sub_agents",
        "hierarchical",
    },
    "parallel": {
        "parallel",
        "broadcast",
        "fan_out",
        "fan_out_fan_in",
        "adk_parallel_worker",
    },
    "fan_out_fan_in": {
        "fan_out_fan_in",
        "parallel",
        "adk_parallel_worker",
    },
    "sequential_dependency": {
        "sequential_dependency",
        "sequential",
        "sequential_task_dependency",
        "sequential_team_dependency",
    },
    "iterative_feedback": {
        "iterative_feedback",
        "feedback_loop",
        "round_robin_group",
    },
    "round_robin_group": {
        "round_robin_group",
        "iterative_feedback",
    },
    "sequential_task_dependency": {
        "sequential_task_dependency",
        "sequential_dependency",
    },
    "publish_subscribe": {
        "publish_subscribe",
        "event_driven",
        "feedback_loop",
    },
    "feedback_loop": {
        "feedback_loop",
        "iterative_feedback",
    },
    "nested_team_composition": {
        "nested_team_composition",
        "composition",
    },
    "hierarchical_coordination": {
        "hierarchical_coordination",
        "hierarchical",
        "coordination",
    },
    "sequential_team_dependency": {
        "sequential_team_dependency",
        "sequential_dependency",
        "sequential",
    },
    "human_in_the_loop": {
        "human_in_the_loop",
        "hitl",
        "human_review",
        "human_review_node",
    },
    "conditional_routing": {
        "conditional_routing",
        "conditional_handoff",
        "conditional_graph",
    },
}


def pattern_satisfied(actual: Set[str], expected: str) -> bool:
    actual = {x.lower() for x in actual}
    allowed = PATTERN_ALIASES.get(
        expected.lower(),
        {expected.lower()},
    )
    return bool(actual & allowed)


# ---------------------------------------------------------------------
# Gold-case locating
# ---------------------------------------------------------------------

def find_case_system(
    repos: List[Dict[str, Any]],
    case_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    expected_names = GOLD_REPOS[case_id]
    expected_files = EXACT_FILES[case_id]

    repo_candidates = [
        r for r in repos
        if repo_matches(get_repo_name(r), expected_names)
    ]

    if not repo_candidates:
        return None, None, "MISSING_REPOSITORY"

    # For GC10, the expected source is a framework class, and the correct
    # result is no application-level system.  Still return the repo.
    repo = repo_candidates[0]
    systems = get_systems(repo)

    for system in systems:
        if system_matches_files(system, expected_files):
            return repo, system, "FOUND_EXACT_SYSTEM"

    if case_id == "GC10":
        return repo, None, "NO_EXACT_SYSTEM_EXPECTED"

    return repo, None, "MISSING_EXACT_SYSTEM"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_case(
    case_id: str,
    repo: Optional[Dict[str, Any]],
    system: Optional[Dict[str, Any]],
    status: str,
    gold: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    exp = gold.get(case_id, {})
    result = {
        "id": case_id,
        "status": "REVIEW",
        "reason": status,
        "repo": get_repo_name(repo) if repo else "",
        "source_files": sorted(source_files(system)) if system else [],
        "actual_agents": len(get_agents(system)) if system else 0,
        "expected_agents": exp["agents"],
        "actual_architecture": get_architecture(system) if system else None,
        "expected_architecture": exp.get("architecture"),
        "actual_specialization": get_specialization(system) if system else None,
        "expected_specialization": exp.get("specialization"),
        "actual_patterns": sorted(get_patterns(system)) if system else [],
        "expected_patterns": sorted(exp["patterns"]),
        "actual_edges": len(get_edges(system)) if system else 0,
        "checks": {},
        "notes": [],
    }

    # ---------------------------------------------------------------
    # Missing repository/system
    # ---------------------------------------------------------------
    if status == "MISSING_REPOSITORY":
        result["status"] = "FAIL"
        result["notes"].append("Gold repository was not present in the V6.6 output.")
        return result

    if status == "MISSING_EXACT_SYSTEM":
        result["status"] = "FAIL"
        result["notes"].append(
            "Repository exists, but no system was attached to the exact gold source file(s)."
        )
        result["notes"].append(
            "This is primarily a system-boundary/source-attachment failure."
        )
        return result

    # ---------------------------------------------------------------
    # Negative control
    # ---------------------------------------------------------------
    if case_id == "GC10":
        actual_systems = len(get_systems(repo))
        ok = actual_systems == 0
        result["checks"]["no_application_system"] = ok

        if ok:
            result["status"] = "PASS"
            result["notes"].append(
                "Framework-internal Agent class did not become an application-level system."
            )
        else:
            result["status"] = "FAIL"
            result["notes"].append(
                f"Expected 0 application systems, found {actual_systems}."
            )
        return result

    # ---------------------------------------------------------------
    # Agent count
    # ---------------------------------------------------------------
    actual_agents = len(get_agents(system))
    count_ok = actual_agents == exp["agents"]
    result["checks"]["agent_count"] = count_ok

    # ---------------------------------------------------------------
    # Interaction evidence
    # ---------------------------------------------------------------
    actual_patterns = get_patterns(system)
    missing_patterns = [
        p for p in exp["patterns"]
        if not pattern_satisfied(actual_patterns, p)
    ]
    patterns_ok = len(missing_patterns) == 0
    result["checks"]["interaction_evidence"] = patterns_ok

    # ---------------------------------------------------------------
    # Edge count is diagnostic, NOT a universal requirement.
    # Some frameworks represent interaction semantically without
    # explicit graph edges (CrewAI, AutoGen, MetaGPT, Agno).
    # ---------------------------------------------------------------
    actual_edges = len(get_edges(system))
    edge_ok = actual_edges >= exp["min_edges"]
    result["checks"]["edge_count"] = edge_ok

    # ---------------------------------------------------------------
    # Architecture:
    # V6.6 intentionally separates evidence extraction from semantic
    # normalization. architecture_primary=None is therefore REVIEW,
    # not an extractor FAIL, if the structural evidence is present.
    # ---------------------------------------------------------------
    actual_arch = get_architecture(system)
    architecture_ok = (
        actual_arch is not None
        and (
            actual_arch == exp["architecture"].lower()
            or (
                exp["architecture"] == "conditional_graph"
                and actual_arch in {"conditional_routing", "graph", "conditional_graph"}
            )
            or (
                exp["architecture"] == "hierarchical_nested_team"
                and actual_arch in {"hierarchical_nested_team", "hierarchical", "nested_team_composition"}
            )
        )
    )
    result["checks"]["architecture_primary"] = architecture_ok

    if actual_arch is None:
        result["notes"].append(
            "architecture_primary is null. V6.6 is intentionally evidence-preserving; "
            "this is a semantic-normalization REVIEW, not automatically a miner failure."
        )

    # ---------------------------------------------------------------
    # Specialization:
    # V6.6 is not expected to infer research-level specialization from
    # raw code evidence alone. Presence is diagnostic only.
    # ---------------------------------------------------------------
    actual_spec = get_specialization(system)
    spec_ok = (
        actual_spec is not None
        and actual_spec == exp["specialization"].lower()
    )
    result["checks"]["specialization"] = spec_ok

    if actual_spec is None:
        result["notes"].append(
            "No specialization label found. Treat this as expected if "
            "specialization is being assigned by a later semantic classifier."
        )

    # ---------------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------------
    if not count_ok:
        result["status"] = "FAIL"
        result["notes"].insert(
            0,
            f"Agent count mismatch: expected {exp['agents']}, got {actual_agents}."
        )
        return result

    if not patterns_ok:
        result["status"] = "FAIL"
        result["notes"].insert(
            0,
            "Required structural interaction evidence is missing: "
            + ", ".join(missing_patterns)
        )
        return result

    # Architecture and specialization can legitimately remain unresolved
    # at the evidence-extraction stage.
    if not architecture_ok or not spec_ok:
        result["status"] = "REVIEW"
        if not architecture_ok and actual_arch is not None:
            result["notes"].append(
                f"Architecture label differs from gold: actual={actual_arch!r}, "
                f"expected={exp['architecture']!r}."
            )
        if not spec_ok and actual_spec is not None:
            result["notes"].append(
                f"Specialization label differs from gold: actual={actual_spec!r}, "
                f"expected={exp['specialization']!r}."
            )
        return result

    result["status"] = "PASS"
    return result


def print_result(r: Dict[str, Any]) -> str:
    checks = r["checks"]
    return (
        f"{r['id']:<4} {r['status']:<6} "
        f"agents={r['actual_agents']}/{r['expected_agents']}  "
        f"edges={r['actual_edges']}  "
        f"arch={r['actual_architecture'] or 'null'}  "
        f"spec={r['actual_specialization'] or 'null'}  "
        f"patterns={','.join(r['actual_patterns']) or '-'}"
    )


def main() -> None:
    global CLASSIFICATIONS_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifications", default=str(CLASSIFICATIONS_FILE),
                    help="Semantic classifications JSON to merge (default: v1).")
    args = ap.parse_args()
    CLASSIFICATIONS_FILE = Path(args.classifications)

    if not EVIDENCE_FILE.exists():
        print(f"ERROR: Missing {EVIDENCE_FILE}")
        print("Run code_miner_v6_6.py first.")
        return

    try:
        evidence = load_json(EVIDENCE_FILE)
    except Exception as e:
        print(f"ERROR: Could not read {EVIDENCE_FILE}: {e}")
        return

    repos = flatten_repo_records(evidence)
    if not repos:
        print("ERROR: No repository records found in the V6.6 evidence JSON.")
        return

    merged = merge_semantic_classifications(repos)
    gold = load_gold()
    results = []

    print("=" * 100)
    print("V6.6 GOLD VALIDATION")
    print("=" * 100)
    print(f"Evidence : {EVIDENCE_FILE}")
    print(f"Gold     : {GOLD_FILE if GOLD_FILE.exists() else 'embedded expectations'}")
    print(f"Semantic : {CLASSIFICATIONS_FILE.name if merged else 'NONE (raw evidence only)'}"
          + (f"  ({merged} systems labeled)" if merged else ""))
    print(f"Repos    : {len(repos)}")
    print()

    for case_id in EXPECTED:
        repo, system, status = find_case_system(repos, case_id)
        result = validate_case(case_id, repo, system, status, gold)
        results.append(result)
        print(print_result(result))

        for note in result["notes"]:
            print(f"      - {note}")

    pass_count = sum(r["status"] == "PASS" for r in results)
    review_count = sum(r["status"] == "REVIEW" for r in results)
    fail_count = sum(r["status"] == "FAIL" for r in results)

    print()
    print("-" * 100)
    print(
        f"SUMMARY: PASS={pass_count}  REVIEW={review_count}  FAIL={fail_count}  "
        f"TOTAL={len(results)}"
    )
    print("-" * 100)

    print("\nInterpretation:")
    print("  PASS   = structural extraction matches the gold case.")
    print("  REVIEW = evidence is present, but semantic normalization/labeling is unresolved.")
    print("  FAIL   = repository/system boundary, agent count, or required structural evidence is wrong.")
    print()
    print("Important: architecture_primary=null and missing specialization labels are")
    print("NOT automatically treated as extractor failures in V6.6.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as f:
        f.write("V6.6 GOLD VALIDATION\n")
        f.write("=" * 100 + "\n")
        f.write(f"Evidence: {EVIDENCE_FILE}\n")
        f.write(f"Gold: {GOLD_FILE if GOLD_FILE.exists() else 'embedded expectations'}\n")
        f.write(f"Semantic: {CLASSIFICATIONS_FILE.name if merged else 'NONE (raw evidence only)'}"
                + (f" ({merged} systems labeled)\n" if merged else "\n"))
        f.write(f"Repositories: {len(repos)}\n\n")

        for r in results:
            f.write(print_result(r) + "\n")
            for note in r["notes"]:
                f.write(f"    - {note}\n")
            f.write("\n")

        f.write("-" * 100 + "\n")
        f.write(
            f"SUMMARY: PASS={pass_count} REVIEW={review_count} FAIL={fail_count} "
            f"TOTAL={len(results)}\n"
        )
        f.flush()
        os.fsync(f.fileno())

    # Atomic swap so an abrupt stop never leaves a truncated report.
    os.replace(tmp_output, OUTPUT_FILE)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
