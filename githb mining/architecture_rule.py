"""
architecture_rule.py

Deterministic precedence rule for architecture_primary, derived from the
structural evidence the V6.6 miner extracts. Single source of truth, imported
by both the hybrid classifier and the validator.

Motivation: V1's LLM over-applied `sequential` (0/3 target gold errors fixed by
prompt iteration; it also invented `sequential` for systems with NO structural
evidence). Architecture is structural, so it is assigned by an explicit,
reproducible precedence over the extracted signals instead of by the LLM.

Precedence (a tie-break when mechanisms coexist, applied to the union of a
system's architecture_patterns + interaction_primitives + architecture_evidence_types):
    parallel > event_driven > hierarchical > conditional_routing
             > iterative_feedback > peer_to_peer > sequential
`sequential` is chosen only when ordering is the sole mechanism present.
A system with 2+ agents but NO structural architecture signal is `unresolved`
(honest: the evidence is insufficient), not silently `sequential`.

Validated 9/9 on the gold architecture cases (GC01-GC09; GC10 has no system).
Bare `composition` is intentionally excluded from the hierarchical set — only
nested_team_composition / delegation / coordination count — so a plain team
wrapper does not become "hierarchical".
"""

PARALLEL = {"parallel", "fan_out_fan_in", "fan_out", "broadcast", "adk_parallel_worker"}
EVENT = {"event_driven", "publish_subscribe", "event_subscription", "producer_consumer"}
HIER = {
    "hierarchical", "hierarchical_delegation", "hierarchical_coordination",
    "nested_team_composition", "sub_agent_delegation",
}
COND = {"conditional_routing", "conditional_handoff", "handoff", "conditional_graph", "routing"}
ITER = {"iterative_feedback", "feedback_loop", "round_robin_group"}
P2P = {"peer_to_peer"}


def signals(system):
    """Union of the structural signals the miner attached to a system/bundle."""
    s = set()
    for key in ("architecture_patterns", "interaction_primitives", "architecture_evidence_types"):
        vals = system.get(key) or []
        if isinstance(vals, list):
            s.update(str(v).strip().lower() for v in vals if v is not None)
    return s


def architecture_primary(system):
    """Return the dominant architecture label by precedence over `signals`."""
    s = signals(system)
    if s & PARALLEL:
        return "parallel"
    if s & EVENT:
        return "event_driven"
    if s & HIER:
        return "hierarchical"
    if s & COND:
        return "conditional_routing"
    if s & ITER:
        return "iterative_feedback"
    if s & P2P:
        return "peer_to_peer"
    if any("sequential" in t for t in s):
        return "sequential"
    try:
        if int(system.get("agent_count") or 0) <= 1:
            return "single_agent"
    except (TypeError, ValueError):
        pass
    return "unresolved"
