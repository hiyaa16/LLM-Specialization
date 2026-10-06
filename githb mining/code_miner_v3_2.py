"""
code_miner_v6_6.py

System-first, evidence-preserving miner for open-source LLM multi-agent systems.

Research rule:
    EXTRACT EVIDENCE FROM CODE; DO NOT DECIDE SPECIALIZATION OR
    ARCHITECTURE_PRIMARY PREMATURELY.

Primary unit:
    An identifiable multi-agent application/example/template/benchmark system.

Design:
    1. Build source/provenance index.
    2. Discover candidate systems from explicit structural anchors.
    3. Extract agent evidence only in/connected to those candidate systems.
    4. Extract structural interaction evidence.
    5. Emit auditable evidence bundles.

V6.6 additionally treats orchestration units as first-class structural
objects for system-boundary discovery. Nested compositions are resolved
recursively without treating an orchestrator as an agent. Architecture labels
remain downstream classification variables.

This version deliberately separates:
    raw detections (diagnostic only)
    from
    system-attached evidence (research denominator).

It does not use repository-specific paths, exact tool names, or README
keywords as proof of an agent/system.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(".")
REPOS_DIR = ROOT / "repos"
DATA_DIR = ROOT / "data"

PY_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
NOTEBOOK_EXTENSIONS = {".ipynb"}
MAX_FILE_BYTES = 2_000_000
# Notebook JSON is bloated by embedded cell outputs (plots, base64 images)
# that have nothing to do with code volume, so the raw-file cap used for
# .py/.js source doesn't apply to it; the actual code-size cap is enforced
# after extracting just the code cells (see notebook_code_source / mine_repo).
MAX_NOTEBOOK_FILE_BYTES = 20_000_000
NOTEBOOK_MAGIC_LINE_RE = re.compile(r"^\s*[!%]")

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    "coverage", ".next", ".nuxt", "out", "dist", "build", "vendor",
    "generated", "gen", "site-packages", ".eggs", "htmlcov",
    "storybook-static", ".idea", ".vscode",
}

TEST_DIR_NAMES = {
    "test", "tests", "__tests__", "testing", "integration_tests",
    "unit_tests", "e2e_tests",
}
DOC_DIR_NAMES = {"docs", "doc", "documentation"}
EXAMPLE_DIR_NAMES = {
    "examples", "example", "samples", "sample", "cookbook", "cookbooks",
    "demos", "demo", "tutorial", "tutorials", "quickstart", "showcase",
}
BENCHMARK_DIR_NAMES = {"benchmarks", "benchmark"}
TEMPLATE_DIR_NAMES = {"templates", "template"}

# Used only to identify repositories whose framework implementation must be
# separated from their examples/cookbooks. This is NOT an application filter.
KNOWN_FRAMEWORK_REPOS = {
    "agentscope-ai/agentscope",
    "microsoft/autogen",
    "foundationagents/metagpt",
    "agno-agi/agno",
    "google/adk-python",
    "openai/openai-agents-python",
    "mervinpraison/praisonai",
}

FRAMEWORK_IMPORTS = {
    "agents": "OpenAI Agents",
    "openai.agents": "OpenAI Agents",
    "agentscope": "AgentScope",
    "autogen": "AutoGen",
    "autogen_agentchat": "AutoGen",
    "crewai": "CrewAI",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "google.adk": "Google ADK",
    "google.genai": "Google ADK",
    "metagpt": "MetaGPT",
    "agno": "Agno",
    "praisonai": "PraisonAI",
    "camel": "CAMEL",
}

AGENT_CONSTRUCTORS = {
    "Agent", "AssistantAgent", "UserProxyAgent", "ConversableAgent",
    "LlmAgent", "LLMAgent", "ReActAgent", "ChatAgent", "RoutedAgent",
    "RolePlayingAgent", "WorkerAgent", "OpenAIAgent", "GeminiAgent",
    "SpecialistAgent", "TaskAgent",
}

ORCHESTRATORS = {
    "SequentialAgent", "ParallelAgent", "LoopAgent", "Workflow", "Graph",
    "StateGraph", "Crew", "Team", "RoundRobinGroupChat",
    "SelectorGroupChat", "Swarm",
}

AGENTISH_BASES = {
    "Agent", "BaseAgent", "AbstractAgent", "Role", "RoleBase",
    "AssistantAgent", "ConversableAgent",
}

BEHAVIOR_METHODS = {
    "run", "arun", "reply", "areply", "respond", "arespond", "execute",
    "aexecute", "chat", "achat", "invoke", "ainvoke", "step", "astep",
    "act", "aact", "observe", "reason", "think", "plan",
}

RESPONSIBILITY_FIELDS = {
    "role", "goal", "backstory", "instructions", "instruction",
    "system_prompt", "system_message", "persona", "description", "objective",
    "purpose", "task", "prompt", "profile",
}

ARCHITECTURAL_TYPES = {
    "sequential", "parallel", "hierarchical", "peer_to_peer",
    "conditional_routing", "iterative_feedback", "event_driven",
    "sequential_pre_post",
}

# Interaction primitives are evidence-level observations. They are broader
# than the final architecture taxonomy and may legitimately remain unresolved.
INTERACTION_PRIMITIVES = ARCHITECTURAL_TYPES | {
    "coordination", "composition", "conditional_handoff",
    "fan_out_fan_in", "sequential_dependency", "hierarchical_delegation",
    "nested_team_composition", "human_in_the_loop", "publish_subscribe",
    "feedback_loop", "round_robin_group", "sequential_task_dependency",
    "hierarchical_coordination", "sequential_team_dependency",
}

CONTROL_NAME_TOKENS = {
    "validate", "validation", "check", "checker", "aggregate", "aggregator",
    "reduce", "merge", "format", "formatter", "transform", "preprocess",
    "postprocess", "interrupt", "route", "router", "should_continue",
    "condition", "conditional", "start", "end", "human_review", "humanreview",
    "hitl", "serialize", "deserialize", "save", "load", "parse", "parser",
}

ROLE_NAME_TOKENS = {
    "research", "researcher", "write", "writer", "review", "reviewer",
    "critic", "planner", "supervisor", "analyst", "analyzer", "executor",
    "worker", "generator", "search", "editor", "coder", "tester", "verifier",
    "manager", "summarizer", "retriever", "classifier", "architect", "designer",
    "assistant", "developer", "explorer", "operator", "specialist",
}

# Framework implementation path patterns. They are intentionally structural
# rather than repository-specific.
FRAMEWORK_INTERNAL_ROOTS = {
    "src", "sdk", "framework", "runtime", "core", "lib", "packages",
    "internal", "engine", "implementation",
}


def norm(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def parts(p: str | Path) -> list[str]:
    return [x.lower() for x in norm(p).split("/") if x]


def repo_name(repo_dir: Path) -> str:
    return repo_dir.name.replace("__", "/", 1)


def short_name(x: str | None) -> str | None:
    return x.split(".")[-1] if x else None


def dotted(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def line(node: ast.AST | None) -> int | None:
    return getattr(node, "lineno", None) if node is not None else None


def segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def known_framework_repo(repo: str) -> bool:
    return repo.lower() in KNOWN_FRAMEWORK_REPOS


def provenance(rel: str, repo: str) -> str:
    ps = parts(rel)
    s = set(ps)

    if s & TEST_DIR_NAMES or any(
        x.startswith("test_") or x.endswith("_test.py") or
        x.endswith(".test.js") or x.endswith(".test.ts")
        for x in ps
    ):
        return "test"
    if s & DOC_DIR_NAMES:
        return "documentation"
    if s & EXAMPLE_DIR_NAMES:
        return "example"
    if s & BENCHMARK_DIR_NAMES:
        return "benchmark"
    if s & TEMPLATE_DIR_NAMES:
        return "template"

    if known_framework_repo(repo):
        # Framework repos are mixed repositories. Exclude implementation trees
        # but preserve examples/cookbooks/samples found above.
        if ps and ps[0] in FRAMEWORK_INTERNAL_ROOTS:
            return "framework_internal"
        if len(ps) >= 2 and ps[:2] in (
            ["python", "src"], ["packages", "core"], ["packages", "src"]
        ):
            return "framework_internal"

    return "application"


def framework_names(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [n.module or ""]
        for m in mods:
            for key, fw in FRAMEWORK_IMPORTS.items():
                if m == key or m.startswith(key + "."):
                    out.add(fw)
    return out


def imported_symbols(tree: ast.AST) -> dict[str, str]:
    """Return import bindings while preserving relative-import information."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                local = a.asname or a.name.split(".")[0]
                out[local] = a.name
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            prefix = "." * int(getattr(n, "level", 0))
            for a in n.names:
                if a.name == "*":
                    continue
                origin = prefix + ((mod + ".") if mod else "") + a.name
                out[a.asname or a.name] = origin
    return out


def build_parent_map(tree: ast.AST) -> dict:
    parent = {}
    for n in ast.walk(tree):
        for child in ast.iter_child_nodes(n):
            parent[child] = n
    return parent


def assignment_target(
    tree: ast.AST, target_node: ast.AST, parent: dict | None = None
) -> str | None:
    """Return the lexical binding for a constructor call when one exists.

    Handles direct assignments, constructor calls nested in container
    literals, and `return Agent(...)` inside a factory method/function (the
    enclosing function name becomes the binding). Nested calls without a
    lexical binding receive no fake variable name; they are represented later
    by a stable inline identity.

    `parent` may be a pre-built parent map (see `build_parent_map`) so callers
    processing many nodes in the same file don't rebuild it every time; if
    omitted, one is built for this call only.
    """
    if parent is None:
        parent = build_parent_map(tree)

    cur = target_node
    while cur in parent:
        par = parent[cur]
        if isinstance(par, ast.Assign) and par.value is cur and len(par.targets) == 1:
            return dotted(par.targets[0])
        if isinstance(par, ast.AnnAssign) and par.value is cur:
            return dotted(par.target)
        if isinstance(par, ast.Return) and par.value is cur:
            fn = parent.get(par)
            while fn is not None and not isinstance(
                fn, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                fn = parent.get(fn)
            return fn.name if fn is not None else None
        # A named element such as {"researcher": Agent(...)} is not a
        # variable binding, so do not invent the dict key as an agent symbol.
        # Only climb through pure literal containers. A `keyword`/`Call`
        # boundary means this node was passed as an ARGUMENT to some call,
        # not assigned to a name — e.g. an Agent() constructed inline inside
        # another Agent's `handoffs=[...]`. Climbing past that boundary would
        # wrongly attribute the inline object's identity to the *outer*
        # call's own assigned variable, colliding two distinct agents.
        if isinstance(par, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            cur = par
            continue
        break
    return None


def resolve_element_list(tree: ast.AST, node: ast.AST | None) -> list[ast.AST]:
    """Return element nodes for a members/agents/handoffs-style kwarg value.

    Handles the direct `[...]`/`(...)` literal case as well as one level of
    local variable indirection, e.g.:

        pipeline_agents = [researcher, writer]
        SequentialAgent(sub_agents=pipeline_agents)

    Only an exact, unambiguous same-file list/tuple assignment is followed.
    This is source-level identity resolution, not a symbolic-execution
    engine, so dynamically built lists (append/extend, conditionals, etc.)
    are intentionally left unresolved.
    """
    if node is None:
        return []
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.Name):
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == node.id for t in n.targets)
                and isinstance(n.value, (ast.List, ast.Tuple))
            ):
                return list(n.value.elts)
    return []


def responsibility_items(items: list[tuple[str, ast.AST]], source: str) -> list[dict]:
    out = []
    for field, node in items:
        val = literal(node)
        text = str(val) if val is not None else segment(source, node).strip()
        if text:
            out.append({
                "field": field,
                "text": text[:4000],
                "line": line(node),
                "evidence_type": "responsibility_field",
            })
    return out


def class_agent_info(node: ast.ClassDef, source: str):
    bases = {short_name(dotted(b)) for b in node.bases}

    # A class that subclasses a known orchestration unit (e.g. a custom
    # `class ContentPipelineAgent(SequentialAgent):`) is itself an
    # orchestrator, never an LLM agent. Name/behavior heuristics below must
    # not override this, or orchestrators get double-counted as agents.
    if bases & ORCHESTRATORS:
        return False, [], ""

    if bases & AGENTISH_BASES or bases & AGENT_CONSTRUCTORS:
        basis = "agent_base"
    else:
        methods = {
            x.name.lower() for x in node.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if ("agent" in node.name.lower() or node.name.lower().endswith("role")) and methods & BEHAVIOR_METHODS:
            basis = "agent_like_name_behavior"
        else:
            return False, [], ""

    resp = []
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            target = child.targets[0] if isinstance(child, ast.Assign) and child.targets else child.target
            field = short_name(dotted(target))
            if field in RESPONSIBILITY_FIELDS and getattr(child, "value", None) is not None:
                val = literal(child.value)
                text = str(val) if val is not None else segment(source, child.value).strip()
                if text:
                    resp.append({
                        "field": field, "text": text[:4000],
                        "line": line(child), "evidence_type": "class_responsibility"
                    })
        elif isinstance(child, ast.Call) and short_name(dotted(child.func)) == "set_actions":
            vals = list(child.args[0].elts) if child.args and isinstance(child.args[0], (ast.List, ast.Tuple, ast.Set)) else list(child.args)
            for v in vals:
                text = dotted(v) or segment(source, v).strip()
                if text:
                    resp.append({
                        "field": "set_actions", "text": text[:4000],
                        "line": line(v), "evidence_type": "role_action"
                    })
    return True, resp, basis


def is_agent_constructor(call: ast.Call, custom_classes: set[str], custom_orchestrators: set[str] = frozenset()):
    name = short_name(dotted(call.func)) or ""
    full = dotted(call.func) or ""

    # Orchestration units are NOT agents. This check must happen before the
    # generic "*Agent" fallback because names such as SequentialAgent also
    # end in "Agent" — and a locally-defined subclass of an orchestrator
    # (e.g. `class ContentPipelineAgent(SequentialAgent)`) is just as much
    # an orchestrator even though its own name also ends in "Agent".
    if name in ORCHESTRATORS or name in custom_orchestrators:
        return False, full, ""

    if name in AGENT_CONSTRUCTORS:
        return True, full, "explicit_constructor"
    if name in custom_classes:
        return True, full, "custom_agent_class"

    # Conservative generic fallback: only a class/function explicitly named
    # *Agent is a candidate. It is never enough by itself to create a system.
    if name.endswith("Agent") and len(name) > 5:
        return True, full, "agent_named_constructor"

    return False, full, ""


def inline_identity(node: ast.AST) -> str:
    """Stable per-source identity for an unassigned inline constructor call.

    Includes column offset, not just line, so multiple inline instances on
    the same source line (e.g. `hire([RoleA(), RoleB(), RoleC()])`) each get
    a distinct identity instead of colliding on one shared symbol. This is
    NEVER a repository-wide symbol and must never be used for cross-file
    matching.
    """
    return f"__inline_agent_line_{line(node)}_{getattr(node, 'col_offset', 0)}"


def make_agent(repo, rel, node, display, ctor, resp, prov, evidence, kind="agent_instance", conf="HIGH"):
    assigned = evidence.get("assigned_symbol")
    if not assigned:
        assigned = inline_identity(node)
    return {
        "repository": repo, "file": norm(rel), "line": line(node),
        "name": display, "symbol": assigned,
        "constructor": ctor, "kind": kind, "responsibilities": resp,
        "provenance": prov, "confidence": conf, "evidence": evidence,
    }


def add_edge(out, repo, rel, node, typ, evidence_type, src=None, dst=None, **extra):
    out.append({
        "repository": repo, "file": norm(rel), "line": line(node),
        "type": typ, "source_agent": src, "target_agent": dst,
        "confidence": "HIGH", "evidence_type": evidence_type, **extra
    })


def extract_py(repo: str, rel: str, source: str):
    prov = provenance(rel, repo)
    if prov in {"test", "documentation", "framework_internal"}:
        return [], [], set(), {}, {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], [], set(), {}, {}

    parent_map = build_parent_map(tree)
    imports = imported_symbols(tree)
    fws = framework_names(tree)
    agents, edges = [], []
    custom = {}
    custom_names = set()
    custom_orchestrator_names = set()

    # Factory-method identity resolution: `obj = SomeImportedClass()` followed
    # by `x = obj.method_name()` is a common pattern (e.g. CrewAI tutorials
    # building agents inside a class's methods). `method_name` becomes the
    # binding for `return Agent(...)` inside that method (see
    # `assignment_target`'s Return handling), so recording which local
    # variables are aliases for "class_origin.method_name" lets cross-file
    # system discovery resolve `x` back to that exact agent identity later.
    instance_class_of_var: dict[str, str] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            continue
        val = n.value
        if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
            origin = imports.get(val.func.id)
            if origin:
                instance_class_of_var[n.targets[0].id] = origin

    factory_refs: dict[str, tuple[str, str]] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            continue
        val = n.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and isinstance(val.func.value, ast.Name)
            and val.func.value.id in instance_class_of_var
        ):
            factory_refs[n.targets[0].id] = (
                instance_class_of_var[val.func.value.id], val.func.attr
            )

    # Human-readable `name=` labels, keyed by the lexical symbol they're
    # assigned to. Orchestrator instructions/prompts refer to members by
    # this display name ("Research Team"), not by their python identifier,
    # so ordering language in that free text can only be matched against it.
    display_name_by_symbol: dict[str, str] = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name_node = next((k.value for k in n.keywords if k.arg == "name"), None)
        display_val = literal(name_node)
        if not isinstance(display_val, str):
            continue
        sym = assignment_target(tree, n, parent_map)
        if sym:
            display_name_by_symbol[sym] = display_val

    # Explicit workflow parallel-worker evidence (e.g. ADK). A decorator such
    # as @node(parallel_worker=True), or a `parallel_worker=True` kwarg on an
    # agent constructor, is a structural workflow signal. Computed up front
    # (rather than in a later pass) so orchestrator-level edge extraction
    # below can tell which chain members are parallel workers.
    parallel_worker_symbols: set[str] = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for dec in fn.decorator_list:
            if not (isinstance(dec, ast.Call) and short_name(dotted(dec.func)) == "node"):
                continue
            if any(k.arg == "parallel_worker" and literal(k.value) is True for k in dec.keywords):
                parallel_worker_symbols.add(fn.name)
                add_edge(edges, repo, rel, dec, "parallel", "adk_parallel_worker",
                         None, None, worker_function=fn.name)

    for call_node in ast.walk(tree):
        if not isinstance(call_node, ast.Call):
            continue
        if not any(k.arg == "parallel_worker" and literal(k.value) is True for k in call_node.keywords):
            continue
        sym = assignment_target(tree, call_node, parent_map) or inline_identity(call_node)
        parallel_worker_symbols.add(sym)
        add_edge(edges, repo, rel, call_node, "parallel", "adk_parallel_worker",
                 None, None, worker_function=sym)

    # Definitions.
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            bases = {short_name(dotted(b)) for b in n.bases}
            if bases & ORCHESTRATORS:
                custom_orchestrator_names.add(n.name)
            ok, resp, basis = class_agent_info(n, source)
            if ok:
                custom[n.name] = (n, resp, basis)
                custom_names.add(n.name)

    # Instances.
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        ok, ctor, basis = is_agent_constructor(n, custom_names, custom_orchestrator_names)
        if not ok:
            continue

        lexical_symbol = assignment_target(tree, n, parent_map)
        # No lexical binding (e.g. an agent constructed inline inside another
        # call's argument list) must NOT collapse to a generic constructor
        # name like "Agent" — that would let unrelated inline instances
        # collide on the same symbol. Use the same stable per-line inline
        # identity that downstream edge resolution (e.g. handoffs=[...])
        # expects for exactly this situation.
        assigned = lexical_symbol or inline_identity(n)
        name_node = next((k.value for k in n.keywords if k.arg == "name"), None)
        display_val = literal(name_node)
        display = str(display_val) if display_val is not None else (lexical_symbol or short_name(ctor) or "agent")

        resp = responsibility_items(
            [(k.arg, k.value) for k in n.keywords if k.arg in RESPONSIBILITY_FIELDS],
            source,
        )
        cref = short_name(ctor)
        if not resp and cref in custom:
            resp = list(custom[cref][1])

        agents.append(make_agent(
            repo, rel, n, display, ctor, resp, prov,
            {
                "assigned_symbol": assigned,
                "constructor": ctor,
                "constructor_basis": basis,
                "class_reference": cref if cref in custom else None,
                "source_line": line(n),
            },
            conf="HIGH" if basis in {"explicit_constructor", "custom_agent_class"} else "MEDIUM",
        ))

        # Explicit handoff.
        for k in n.keywords:
            if k.arg != "handoffs":
                continue
            vals = resolve_element_list(tree, k.value) or [k.value]
            for v in vals:
                target_node = v
                if isinstance(v, ast.Call) and short_name(dotted(v.func)) == "handoff" and v.args:
                    target_node = v.args[0]

                target = None
                if isinstance(target_node, ast.Call):
                    # A third/Nth agent constructed inline directly inside the
                    # handoffs list (no lexical binding). Use the same stable
                    # inline identity that this Call receives when it is
                    # independently recorded as an agent instance, so the
                    # handoff edge and the agent record refer to one symbol.
                    inline_symbol = assignment_target(tree, target_node, parent_map)
                    target = inline_symbol or inline_identity(target_node)
                else:
                    target = dotted(target_node)

                if target:
                    add_edge(edges, repo, rel, v, "conditional_routing",
                             "conditional_handoff", assigned, target)

        # Explicit ADK-style composition.
        for k in n.keywords:
            if k.arg != "sub_agents":
                continue
            vals = resolve_element_list(tree, k.value) or [k.value]
            for v in vals:
                target = dotted(v)
                if target:
                    add_edge(edges, repo, rel, v, "hierarchical",
                             "hierarchical_delegation", assigned, target)

    # Custom role classes (MetaGPT-style).
    for cname, (node, resp, basis) in custom.items():
        has_instance = any(
            a["evidence"].get("class_reference") == cname for a in agents
        )
        if not has_instance:
            agents.append(make_agent(
                repo, rel, node, cname, "Role", resp, prov,
                {"class_name": cname, "agent_class_basis": basis},
                kind="agent_definition", conf="HIGH" if basis == "agent_base" else "MEDIUM",
            ))

        for child in ast.walk(node):
            if isinstance(child, ast.Call) and short_name(dotted(child.func)) == "_watch":
                watched = []
                if child.args:
                    vals = list(child.args[0].elts) if isinstance(child.args[0], (ast.List, ast.Tuple, ast.Set)) else [child.args[0]]
                    watched = [dotted(v) or segment(source, v).strip() for v in vals]
                add_edge(edges, repo, rel, child, "event_driven",
                         "event_subscription", cname, None,
                         watched_events=[x for x in watched if x])

    # MetaGPT-style ordered team hiring. `team.hire([RoleA(), RoleB(), ...])`
    # is direct source evidence of an intended execution pipeline: roles are
    # commonly wired to act in hire order via their own _watch() on the
    # previous role's output message type. The ordered member list itself is
    # observable sequential-dependency evidence, independent of whether the
    # publish/subscribe wiring was also detected.
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if short_name(dotted(n.func)) != "hire" or not n.args:
            continue
        member_nodes = resolve_element_list(tree, n.args[0])
        if len(member_nodes) < 2:
            continue
        member_symbols = []
        for v in member_nodes:
            if isinstance(v, ast.Call):
                sym = assignment_target(tree, v, parent_map) or inline_identity(v)
            else:
                sym = dotted(v)
            if sym:
                member_symbols.append(sym)
        for a, b in zip(member_symbols, member_symbols[1:]):
            add_edge(edges, repo, rel, n, "sequential",
                     "metagpt_hire_sequence", a, b)

    # Explicit CrewAI task dependencies. `context=[task_a, task_b]` is
    # direct source evidence that the current task depends on those tasks.
    # Keep task identities separate from agent identities.
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if (short_name(dotted(n.func)) or "") != "Task":
            continue

        task_symbol = assignment_target(tree, n, parent_map)
        if not task_symbol:
            continue

        for k in n.keywords:
            if k.arg != "context":
                continue
            vals = resolve_element_list(tree, k.value) or [k.value]
            deps = [dotted(v) for v in vals if dotted(v)]
            for dep in deps:
                add_edge(
                    edges, repo, rel, n, "sequential",
                    "sequential_task_dependency",
                    None, None,
                    task=task_symbol,
                    depends_on=dep,
                )

    # Explicit orchestrators/compositions.
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        ctor = short_name(dotted(n.func)) or ""

        if ctor == "Crew":
            process = ""
            members = []
            for k in n.keywords:
                if k.arg == "process":
                    process = segment(source, k.value).lower()
                elif k.arg == "agents":
                    members = [dotted(v) for v in resolve_element_list(tree, k.value) if dotted(v)]
            crew_symbol = assignment_target(tree, n, parent_map)
            if "hierarchical" in process:
                add_edge(edges, repo, rel, n, "hierarchical",
                         "hierarchical_delegation", members=members,
                         orchestrator="Crew", orchestrator_symbol=crew_symbol)
            elif "sequential" in process:
                add_edge(edges, repo, rel, n, "sequential",
                         "crewai_process_sequential", members=members,
                         orchestrator="Crew", orchestrator_symbol=crew_symbol)

        if ctor == "Workflow":
            # ADK-style linear pipeline: Workflow(edges=[(a, b, c, ...)]).
            # A node adjacent to a known parallel_worker is the sequential
            # wrapping around that parallel section (pre/post), not a plain
            # sequential step — kept as a distinct primitive so it isn't
            # confused with an ordinary fully-sequential pipeline.
            for k in n.keywords:
                if k.arg != "edges":
                    continue
                chains = k.value
                if not isinstance(chains, (ast.List, ast.Tuple)):
                    continue
                for chain in chains.elts:
                    if not isinstance(chain, (ast.List, ast.Tuple)):
                        continue
                    node_syms = []
                    for el in chain.elts:
                        sym = dotted(el)
                        if sym is None:
                            val = literal(el)
                            if val is not None:
                                sym = str(val)
                        if sym:
                            node_syms.append(sym)
                    for a, b in zip(node_syms, node_syms[1:]):
                        if a in parallel_worker_symbols or b in parallel_worker_symbols:
                            add_edge(edges, repo, rel, n, "sequential_pre_post",
                                     "adk_workflow_pre_post_parallel", a, b)
                        else:
                            add_edge(edges, repo, rel, n, "sequential",
                                     "adk_workflow_sequential_edge", a, b)

        if ctor == "RoundRobinGroupChat":
            add_edge(edges, repo, rel, n, "iterative_feedback", "round_robin_group")
        elif ctor == "SelectorGroupChat":
            add_edge(edges, repo, rel, n, "conditional_routing", "autogen_selector_group_chat")

        if ctor in ORCHESTRATORS:
            for k in n.keywords:
                if k.arg not in {"sub_agents", "agents", "participants", "members"}:
                    continue
                elts = resolve_element_list(tree, k.value)
                if not elts:
                    continue
                members = [dotted(v) for v in elts if dotted(v)]
                if len(members) < 2:
                    continue

                typ = "composition"
                evidence_type = "orchestrator_membership"
                mode_value = None

                if ctor == "SequentialAgent":
                    typ = "sequential"
                elif ctor == "ParallelAgent":
                    typ = "parallel"
                elif ctor == "LoopAgent":
                    typ = "iterative_feedback"
                elif ctor == "Team":
                    # Team membership alone establishes composition, NOT a
                    # hierarchical architecture.  Agno Teams support several
                    # coordination modes, so only an explicit mode/behavior
                    # should determine the architectural type.
                    for kw in n.keywords:
                        if kw.arg == "mode":
                            mode_value = segment(source, kw.value).strip().lower()
                            break

                    if mode_value:
                        if "broadcast" in mode_value:
                            typ = "parallel"
                            evidence_type = "agno_team_mode_broadcast"
                        elif "route" in mode_value:
                            typ = "conditional_routing"
                            evidence_type = "agno_team_mode_route"
                        elif "coordinate" in mode_value:
                            # Coordination is an explicit interaction primitive,
                            # but does NOT by itself prove hierarchy or peer-to-peer.
                            # Keep architecture_primary unresolved until the
                            # structural evidence distinguishes the topology.
                            typ = "coordination"
                            evidence_type = "agno_team_mode_coordinate"
                        elif "task" in mode_value:
                            typ = "hierarchical"
                            evidence_type = "agno_team_mode_tasks"
                        else:
                            typ = "composition"
                            evidence_type = "agno_team_mode_unknown"
                    elif "Agno" in fws:
                        # Legacy Agno flags are explicit behavioral evidence.
                        # Do NOT infer an architecture from Team's framework
                        # default when the source does not explicitly specify
                        # coordination semantics. The research variable must
                        # be grounded in observable source evidence.
                        flags = {k.arg: literal(k.value) for k in n.keywords
                                 if k.arg in {"respond_directly", "delegate_to_all_members"}}
                        if flags.get("delegate_to_all_members") is True:
                            typ = "parallel"
                            evidence_type = "agno_legacy_delegate_to_all_members"
                        elif flags.get("respond_directly") is True:
                            typ = "conditional_routing"
                            evidence_type = "agno_legacy_respond_directly"
                        else:
                            # Membership is retained as composition evidence,
                            # but architecture remains unresolved.
                            typ = "composition"
                            evidence_type = "agno_team_membership_only"

                # Nested orchestration is a topology fact: a member whose
                # symbol is itself an orchestration object creates an explicit
                # nested-composition primitive. This does not assume hierarchy.
                nested_names = []
                for v in elts:
                    sym = dotted(v)
                    if sym:
                        nested_names.append(sym)

                # Identify whether a listed member is itself an orchestration
                # binding in this same source file. This is structural nesting,
                # not a framework-default hierarchy assumption.
                orchestrator_bindings = set()
                for x in ast.walk(tree):
                    if isinstance(x, (ast.Assign, ast.AnnAssign)) and getattr(x, "value", None) is not None:
                        value = x.value
                        if isinstance(value, ast.Call) and (short_name(dotted(value.func)) in ORCHESTRATORS):
                            target = x.targets[0] if isinstance(x, ast.Assign) and x.targets else x.target
                            sym = dotted(target)
                            if sym:
                                orchestrator_bindings.add(sym)

                evidence_type_final = evidence_type
                if nested_names and any(sym in orchestrator_bindings for sym in nested_names):
                    evidence_type_final = "nested_team_composition"

                orchestrator_symbol = assignment_target(tree, n, parent_map)
                add_edge(edges, repo, rel, n, typ,
                         evidence_type_final, None, None,
                         members=members, orchestrator=ctor,
                         orchestrator_symbol=orchestrator_symbol,
                         mode=mode_value, nested_members=nested_names)

                # Literal free-text evidence of execution order. An
                # orchestrator's own instructions/description sometimes
                # state the intended order directly (e.g. "ask Research Team
                # first, then Writing Team") — that is source-level
                # evidence, not an inferred framework default, so it is
                # matched against the members' own literal `name=` labels
                # (the only names such prose would actually use).
                text_kw = next(
                    (k.value for k in n.keywords if k.arg in {"instructions", "description", "goal"}),
                    None,
                )
                if text_kw is not None:
                    text_val = literal(text_kw)
                    if isinstance(text_val, (list, tuple)):
                        order_text = " ".join(str(x) for x in text_val if isinstance(x, str)).lower()
                    elif isinstance(text_val, str):
                        order_text = text_val.lower()
                    else:
                        order_text = ""
                    if order_text:
                        for a in members:
                            da = display_name_by_symbol.get(a, "").lower()
                            if not da:
                                continue
                            pa = order_text.find(da)
                            if pa == -1:
                                continue
                            for b in members:
                                if b == a:
                                    continue
                                db = display_name_by_symbol.get(b, "").lower()
                                if not db:
                                    continue
                                pb = order_text.find(db)
                                if pb == -1 or pb <= pa:
                                    continue
                                between = order_text[pa + len(da):pb]
                                if re.search(r"\b(then|first|before|after)\b", between):
                                    add_edge(edges, repo, rel, n, "sequential",
                                             "orchestrator_instruction_sequence", a, b)

        # Generic direct dependency patterns commonly used by application code.
        # Only named-agent-looking calls are captured; generic function calls are
        # not interpreted as agent interactions.
        fn = short_name(dotted(n.func))
        if fn in {"handoff", "send_to", "delegate_to", "transfer_to"} and n.args:
            target = dotted(n.args[0])
            if target:
                add_edge(edges, repo, rel, n, "conditional_routing",
                         "explicit_delegation_call", None, target)

    # LangGraph nodes.
    functions = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def graph_node_agent(fn):
        lname = fn.name.lower()
        toks = set(re.findall(r"[a-z0-9]+", lname))
        if toks & CONTROL_NAME_TOKENS:
            return False, [], {}

        role_signal = bool(toks & ROLE_NAME_TOKENS) or "agent" in toks
        doc = ast.get_docstring(fn)
        resp = []
        if doc:
            resp.append({
                "field": "docstring", "text": doc[:4000],
                "line": line(fn), "evidence_type": "function_role_evidence"
            })

        body = segment(source, fn).lower()
        # Strong model/agent creation or LLM call patterns.
        model_constructor = bool(re.search(
            r"\b(chatopenai|chatanthropic|chatgooglegenerativeai|"
            r"azurechatopenai|init_chat_model|create_react_agent|"
            r"create_tool_calling_agent|assistantagent|llmagent|reactagent)\s*\(",
            body
        ))
        model_factory = bool(re.search(r"\b(get_llm|get_model|build_llm|make_llm|load_model)\s*\(", body))
        explicit_agent_call = bool(re.search(r"\b(agent|assistant)\s*\.", body))
        tool_call = False
        tool_symbol = None
        for local, origin in imports.items():
            if re.search(rf"\b{re.escape(local)}\s*\.\s*(?:a?invoke|a?run|execute|call|acall)\s*\(", body):
                tail = origin.lower().split(".")[-1]
                if ".tools." in origin.lower() or "tool" in tail or tail in {"search", "retriever"}:
                    tool_call, tool_symbol = True, local
                    break

        # Role function + imported tool is enough for a specialist node; the
        # tool itself is not hard-coded.
        is_agent = model_constructor or model_factory or explicit_agent_call or (role_signal and tool_call)
        return is_agent, resp, {
            "role_name_signal": role_signal,
            "model_constructor": model_constructor,
            "model_factory": model_factory,
            "explicit_agent_call": explicit_agent_call,
            "imported_tool_call": tool_call,
            "imported_tool_symbol": tool_symbol,
        }

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or short_name(dotted(n.func)) != "add_node":
            continue
        if len(n.args) < 2:
            continue
        node_name = literal(n.args[0])
        fn_name = short_name(dotted(n.args[1]))
        if not node_name or fn_name not in functions:
            continue
        fn = functions[fn_name]
        ok, resp, ev = graph_node_agent(fn)
        if ok:
            agents.append({
                "repository": repo, "file": norm(rel), "line": line(fn),
                "name": str(node_name), "symbol": str(node_name),
                "constructor": "LangGraphNode", "kind": "agent_instance",
                "responsibilities": resp, "provenance": prov,
                "confidence": "HIGH" if resp or ev["model_factory"] or ev["model_constructor"] else "MEDIUM",
                "evidence": {
                    "graph_node": str(node_name), "function": fn_name,
                    "evidence_type": "registered_graph_node", **ev,
                }
            })

    # Explicit human-in-the-loop evidence. These are control nodes, not
    # agents. We only record explicit source signals.
    for fn in functions.values():
        lname = fn.name.lower()
        body = segment(source, fn).lower()

        if (
            "human_review" in lname
            or "humanreview" in lname
            or "hitl" in lname
            or "interrupt(" in body
        ):
            add_edge(
                edges, repo, rel, fn, "conditional_routing",
                "human_in_the_loop",
                None, None,
                control_function=fn.name,
            )

    # LangGraph explicit graph structure.
    def graph_endpoint(node):
        # Prefer a literal string node name; fall back to the dotted name for
        # sentinel references such as the imported START/END constants,
        # which are not string literals but are still explicit structural
        # endpoints of the graph.
        val = literal(node)
        if val is not None:
            return str(val)
        return dotted(node)

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = short_name(dotted(n.func))
        if fn == "add_edge" and len(n.args) >= 2:
            a, b = graph_endpoint(n.args[0]), graph_endpoint(n.args[1])
            if a is not None and b is not None:
                add_edge(edges, repo, rel, n, "sequential",
                         "langgraph_add_edge", a, b)
        elif fn == "add_conditional_edges":
            src = graph_endpoint(n.args[0]) if n.args else None
            add_edge(edges, repo, rel, n, "conditional_routing",
                     "langgraph_add_conditional_edges", src)

    # Evidence for explicit review/revision feedback loops in graph workflows.
    role_names = {
        fn.name.lower()
        for fn in functions.values()
        if set(re.findall(r"[a-z0-9]+", fn.name.lower())) & ROLE_NAME_TOKENS
    }
    has_reviewer = any(
        any(tok in name for tok in {"review", "critic", "verif"})
        for name in role_names
    )
    has_writer = any(
        any(tok in name for tok in {"write", "writer", "generate", "draft"})
        for name in role_names
    )
    conditional_graph = any(
        e.get("evidence_type") == "langgraph_add_conditional_edges"
        for e in edges
    )
    body_lower = source.lower()
    has_revision_signal = bool(
        re.search(r"\b(revise|revision|retry|reject|again|redo)\b", body_lower)
    )

    if has_reviewer and conditional_graph and (has_writer or has_revision_signal):
        add_edge(
            edges, repo, rel,
            next(iter(functions.values())) if functions else tree,
            "iterative_feedback",
            "explicit_review_revision_loop",
            None, None,
        )

    return agents, edges, fws, imports, factory_refs


def extract_js(repo: str, rel: str, source: str):
    prov = provenance(rel, repo)
    if prov in {"test", "documentation", "framework_internal"}:
        return [], [], set(), {}, {}

    agents, edges, fws, imports = [], [], set(), {}
    for m in re.finditer(r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\))", source):
        mod = m.group(1) or m.group(2) or ""
        for key, fw in FRAMEWORK_IMPORTS.items():
            if mod == key or mod.startswith(key + "/"):
                fws.add(fw)

    constructors = "|".join(sorted(map(re.escape, AGENT_CONSTRUCTORS), key=len, reverse=True))
    for m in re.finditer(rf"\b({constructors})\s*\(", source):
        ctor = m.group(1)
        before = source[max(0, m.start()-350):m.start()]
        mm = re.search(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*$", before)
        name = mm.group(1) if mm else ctor
        window = source[m.end():m.end()+5000]
        resp = []
        for field in RESPONSIBILITY_FIELDS:
            x = re.search(rf"\b{re.escape(field)}\s*:\s*(['\"`])(.+?)\1", window, re.DOTALL)
            if x:
                resp.append({
                    "field": field, "text": x.group(2)[:4000],
                    "line": source.count("\n", 0, m.start()) + 1,
                    "evidence_type": "responsibility_field"
                })
        agents.append({
            "repository": repo, "file": norm(rel),
            "line": source.count("\n", 0, m.start()) + 1,
            "name": name, "symbol": name, "constructor": ctor,
            "kind": "agent_instance", "responsibilities": resp,
            "provenance": prov, "confidence": "HIGH",
            "evidence": {"constructor": ctor, "assigned_symbol": name},
        })

    for m in re.finditer(r"\bhandoff\s*\(\s*([A-Za-z_$][\w$]*)", source):
        add_edge(edges, repo, rel, _FakeNode(m, source), "conditional_routing",
                 "explicit_handoff", None, m.group(1))

    for m in re.finditer(r"\.(?:addEdge)\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", source):
        add_edge(edges, repo, rel, _FakeNode(m, source), "sequential",
                 "explicit_edge", m.group(1), m.group(2))

    return agents, edges, fws, imports, {}


class _FakeNode:
    def __init__(self, match, source):
        self.lineno = source.count("\n", 0, match.start()) + 1


def notebook_code_source(nb_text: str) -> str:
    """Concatenate a Jupyter notebook's Python code cells into a single
    pseudo source blob so the existing AST extractor can run over notebook
    tutorials/examples unchanged. Line numbers in resulting evidence are
    relative to this synthesized blob, not real notebook cell/line
    coordinates — an acceptable approximation given evidence is already
    file-relative rather than byte-exact elsewhere in this pipeline.
    """
    try:
        nb = json.loads(nb_text)
    except Exception:
        return ""

    if not isinstance(nb, dict):
        return ""

    lang = ((nb.get("metadata") or {}).get("language_info") or {}).get("name", "")
    if lang and lang.lower() not in {"python", "python3", "ipython"}:
        return ""

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        return ""

    blocks = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src or "")
        if not text.strip():
            continue

        lines = text.split("\n")
        # A `%%something` cell magic (e.g. %%bash, %%writefile) makes the
        # entire cell body non-Python with no reliable way to tell which
        # such directives merely wrap valid Python, so the whole cell is
        # skipped rather than risk corrupting the parse.
        first_nonblank = next((l for l in lines if l.strip()), "")
        if first_nonblank.lstrip().startswith("%%"):
            continue

        cleaned = "\n".join(
            ("# " + l) if NOTEBOOK_MAGIC_LINE_RE.match(l) else l
            for l in lines
        )
        blocks.append(cleaned)

    if not blocks:
        return ""

    whole = "\n\n".join(blocks)
    try:
        ast.parse(whole)
        return whole
    except SyntaxError:
        pass

    # One malformed cell shouldn't blank out an entire notebook's evidence:
    # fall back to keeping only the cells that parse independently.
    valid_blocks = []
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            continue
        valid_blocks.append(block)

    return "\n\n".join(valid_blocks)


def dedupe_agents(items):
    seen, out = set(), []
    for a in items:
        key = (a.get("file"), a.get("line"), a.get("symbol"), a.get("kind"), a.get("constructor"))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def dedupe_edges(items):
    seen, out = set(), []
    for e in items:
        key = (
            e.get("file"), e.get("line"), e.get("type"), e.get("evidence_type"),
            e.get("source_agent"), e.get("target_agent"),
            tuple(e.get("members") or []),
        )
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def agent_key(a):
    return (a.get("file"), a.get("line"), a.get("symbol"), a.get("kind"), a.get("constructor"))


def agent_symbol(a):
    return str(a.get("symbol") or a.get("name") or "")


def edge_symbols(e):
    out = set()
    for k in ("source_agent", "target_agent"):
        if e.get(k):
            out.add(str(e[k]))
    for x in e.get("members") or []:
        out.add(str(x))
    return {x for x in out if x}


def local_agent_symbol(a):
    """Return only the lexical symbol, never a display name."""
    return str(a.get("symbol") or "")


def module_candidates(repo_file: str, module: str, level: int, records_by_file: dict):
    """Resolve an ImportFrom module to repository-relative files.

    This deliberately uses the importing file's package context and exact
    module paths; it does not resolve by basename alone.
    """
    f = norm(repo_file)
    parts = f.split("/")[:-1]
    if level:
        # level=1 means current package; level=2 means parent package, etc.
        keep = max(0, len(parts) - (level - 1))
        base_parts = parts[:keep]
        if module:
            base_parts += module.split(".")
    else:
        base_parts = module.split(".") if module else []
    base = "/".join(x for x in base_parts if x)
    candidates = []
    for ext in (".py", "/__init__.py"):
        cand = base + ext if ext.startswith(".") else base + ext
        if cand in records_by_file:
            candidates.append(cand)
    return candidates


def resolve_dotted_origin(anchor_file, origin, records_by_file):
    """Resolve an absolute dotted `module.attr` origin to candidate files.

    Shared by absolute-import resolution and factory-method resolution
    (where the origin comes from an `imported_symbols` lookup made at
    extraction time rather than from a `from x import y` statement here).
    """
    parts = origin.split(".")
    if len(parts) < 2:
        return [], None
    target_module = "/".join(parts[:-1])
    attr = parts[-1]

    candidates = [
        cand for cand in (target_module + ".py", target_module + "/__init__.py")
        if cand in records_by_file
    ]

    if not candidates:
        # Absolute-looking import in a script/example that is not actually
        # installed as a package: treat it as a sibling of the importing
        # file. This is an exact, same-directory match — not a basename
        # search across the whole repository.
        anchor_dir_parts = norm(anchor_file).split("/")[:-1]
        sibling_module = "/".join(anchor_dir_parts + target_module.split("/"))
        candidates = [
            cand for cand in (sibling_module + ".py", sibling_module + "/__init__.py")
            if cand in records_by_file
        ]

    return candidates, attr


def resolve_import_target(anchor_file, local_name, imported_symbols, records_by_file):
    """Resolve a locally-bound imported name to (candidate files, attr).

    Resolution is module-path based. Basename-only matching is forbidden.
    Relative imports are resolved from the importing file's package context.
    This is shared identity-resolution used both for imported agents and for
    imported orchestration objects (e.g. a Team/Crew built in one file and
    imported into another), so nested composition can be expanded across
    file boundaries with the same exactness guarantees as agent resolution.
    """
    origin = imported_symbols.get(local_name)
    if not origin:
        return [], None

    relative = origin.startswith(".")
    if not relative:
        return resolve_dotted_origin(anchor_file, origin, records_by_file)

    dots = len(origin) - len(origin.lstrip("."))
    rest = origin[dots:]
    parts = norm(anchor_file).split("/")[:-1]
    # `from .x import y` resolves x relative to the current package.
    # `from ..x import y` moves one package upward first.
    up = max(0, dots - 1)
    if up:
        parts = parts[:max(0, len(parts) - up)]
    if "." in rest:
        module, attr = rest.rsplit(".", 1)
        module_parts = parts + module.split(".")
    else:
        # `from . import writer` — writer may itself be a module/object;
        # there is no safe cross-file identity to infer here.
        return [], None
    target_module = "/".join(x for x in module_parts if x)

    candidates = [
        cand for cand in (target_module + ".py", target_module + "/__init__.py")
        if cand in records_by_file
    ]
    return candidates, attr


def resolve_imported_agent(anchor_file, local_name, imported_symbols, records_by_file, agents_by_file_symbol):
    """Resolve an imported local name to exact agent identities."""
    candidates, attr = resolve_import_target(anchor_file, local_name, imported_symbols, records_by_file)
    if not candidates or attr is None:
        return []
    matches = []
    for f in candidates:
        matches.extend(agents_by_file_symbol.get((f, attr), []))
    return matches


def resolve_factory_agent(anchor_file, local_name, records_by_file, agents_by_file_symbol):
    """Resolve `x = obj.method_name()` where `obj` is an instance of an
    imported class, back to the agent(s) that `method_name` constructs.

    Uses the `factory_refs` recorded at extraction time (see `extract_py`):
    a mapping of local variable -> (class_origin, method_name).
    """
    factory_refs = (records_by_file.get(anchor_file) or {}).get("factory_refs") or {}
    ref = factory_refs.get(local_name)
    if not ref:
        return []
    class_origin, method_name = ref
    candidates, _attr = resolve_dotted_origin(anchor_file, class_origin, records_by_file)
    matches = []
    for f in candidates:
        matches.extend(agents_by_file_symbol.get((f, method_name), []))
    return matches

def canonical_agents(agents):
    agents = dedupe_agents(agents)
    instances = {(a["file"], agent_symbol(a)) for a in agents if a["kind"] == "agent_instance"}
    out = []
    for a in agents:
        if a["kind"] == "agent_definition":
            cls = a.get("evidence", {}).get("class_name") or a.get("name")
            if (a["file"], str(cls)) in instances:
                continue
        out.append(a)
    return out


def build_file_graph(source_records):
    # File graph is only for explicit Python imports. It prevents a system
    # anchor in main.py from pulling an unrelated file just because it exists.
    by_base = defaultdict(list)
    for r in source_records:
        p = norm(r["file"])
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        by_base[stem].append(p)

    graph = defaultdict(set)
    for r in source_records:
        f = r["file"]
        for _, origin in r.get("imported_symbols", {}).items():
            base = origin.split(".")[-1]
            for target in by_base.get(base, []):
                if target != f:
                    graph[f].add(target)
                    graph[target].add(f)
    return graph


def structural_edge(e):
    return e.get("type") in ARCHITECTURAL_TYPES or e.get("evidence_type") in {
        "orchestrator_membership", "agno_team_mode_broadcast",
        "agno_team_mode_route", "agno_team_mode_coordinate",
        "agno_team_mode_tasks", "agno_team_mode_unknown",
        "agno_legacy_delegate_to_all_members", "agno_legacy_respond_directly",
        "agno_team_default_coordinate", "agno_team_membership_only",
        "explicit_handoff", "conditional_handoff",
        "explicit_sub_agent_delegation", "hierarchical_delegation",
        "event_subscription", "round_robin_group",
        "sequential_task_dependency", "adk_parallel_worker",
        "human_in_the_loop", "explicit_review_revision_loop",
        "langgraph_add_edge", "langgraph_add_conditional_edges",
        "autogen_round_robin", "autogen_selector_group_chat",
        "crewai_process_hierarchical", "crewai_process_sequential",
        "nested_team_composition",
        "crewai_task_context_dependency",
        "conditional_handoff",
        "hierarchical_delegation",
        "round_robin_group",
        "sequential_task_dependency",
        "adk_parallel_worker",
        "human_in_the_loop",
        "explicit_review_revision_loop",
    }


def discover_systems(repo, records, raw_agents, raw_edges):
    """Discover bounded systems using source-level identity only.

    Evidence extraction rules:
      * same-file lexical identity is always safe;
      * imported symbols are resolved by exact module path + symbol;
      * orchestration units (Crew/Team/SequentialAgent/etc.) are structural
        objects, not agents, and can be recursively expanded when their exact
        members are known;
      * display names and repository-wide symbol matches are never used;
      * unresolved references are retained rather than guessed.

    This function discovers candidate systems. It deliberately does NOT decide
    specialization or architecture_primary.
    """
    eligible = {"application", "example", "template", "benchmark"}
    eligible_files = {r["file"] for r in records if r["provenance"] in eligible}

    def internal_or_low_value(path):
        p = norm(path).lower()
        return (
            "/tests/" in "/" + p + "/" or p.startswith("tests/") or
            p.startswith("test/") or "/test/" in "/" + p + "/" or
            "/src/agentscope/" in "/" + p + "/" or
            "/src/praisonai/" in "/" + p + "/" or
            "/src/autogen/" in "/" + p + "/" or
            "/src/openai_agents/" in "/" + p + "/" or
            "/src/agno/" in "/" + p + "/" or
            "/libs/agno/" in "/" + p + "/" or
            "/metagpt/roles/" in "/" + p + "/"
        )

    agents = [
        a for a in canonical_agents(raw_agents)
        if a["file"] in eligible_files and not internal_or_low_value(a["file"])
    ]
    edges = [
        e for e in dedupe_edges(raw_edges)
        if e["file"] in eligible_files and not internal_or_low_value(e["file"])
        and structural_edge(e)
    ]

    by_file = defaultdict(list)
    by_file_symbol = defaultdict(list)
    for a in agents:
        by_file[a["file"]].append(a)
        by_file_symbol[(a["file"], local_agent_symbol(a))].append(a)

    records_by_file = {r["file"]: r for r in records}

    def resolve_symbol(anchor, sym, imported, seen=None):
        """Resolve a symbol to agent identities or nested composition members."""
        seen = set() if seen is None else set(seen)
        token = (anchor, str(sym))
        if token in seen:
            return [], [{"symbol": str(sym), "reason": "cyclic_or_repeated_reference"}]
        seen.add(token)

        direct = by_file_symbol.get((anchor, str(sym)), [])
        if direct:
            return list(direct), []

        matches = resolve_imported_agent(
            anchor, str(sym), imported, records_by_file, by_file_symbol
        )
        if matches:
            return list(matches), []

        # A local var bound to `instance.factory_method()` where `instance`
        # is built from an imported class (e.g. CrewAI's
        # `editor = agents.editor_agent()`) is not itself an import, so it
        # falls through the check above. Resolve it via the recorded
        # factory_refs identity instead.
        factory_matches = resolve_factory_agent(anchor, str(sym), records_by_file, by_file_symbol)
        if factory_matches:
            return list(factory_matches), []
        # An imported symbol may itself be a nested orchestration object
        # (a Team/Crew built in one file and imported into another) rather
        # than a directly-resolved agent. Fall through to composition
        # expansion so that reference is still exercised via expand_members.

        return [], [{
            "symbol": str(sym),
            "origin": imported.get(str(sym)),
            "reason": "symbol_not_resolved_to_agent"
        }]

    # Build an exact, file-local composition index.  Each composition edge is
    # already evidence extracted from source; here we only connect the named
    # members to their agent identities.  Nested orchestrators are represented
    # by their symbols and can be expanded recursively where the same-file
    # membership evidence exists.
    composition_by_file = defaultdict(lambda: defaultdict(list))
    for e in edges:
        members = [str(x) for x in (e.get("members") or []) if x]
        if not members:
            continue
        # Only composition/orchestrator evidence defines a member set.
        if e.get("evidence_type") in {
            "orchestrator_membership", "agno_team_mode_broadcast",
            "agno_team_mode_route", "agno_team_mode_coordinate",
            "agno_team_mode_tasks", "agno_team_mode_unknown",
            "agno_legacy_delegate_to_all_members", "agno_legacy_respond_directly",
            "agno_team_membership_only", "hierarchical_delegation",
            "crewai_process_sequential",
        }:
            # If a source edge has an explicit source symbol, use it as the
            # composition object identity. Otherwise retain the edge under a
            # synthetic local key so it can still anchor a system.
            owner = str(e.get("source_agent") or
                        e.get("orchestrator_symbol") or
                        f"__composition_line_{e.get('line')}")
            composition_by_file[e["file"]][owner].extend(members)

    cross_file_nested_composition_hits = set()

    def expand_members(anchor, symbols, imported, seen=None):
        seen = set() if seen is None else set(seen)
        found = {}
        unresolved = []
        for sym in symbols:
            key = (anchor, str(sym))
            if key in seen:
                continue
            seen.add(key)

            direct = by_file_symbol.get((anchor, str(sym)), [])
            if direct:
                for a in direct:
                    found[agent_key(a)] = a
                continue

            imported_matches = resolve_imported_agent(
                anchor, str(sym), imported, records_by_file, by_file_symbol
            )
            if imported_matches:
                for a in imported_matches:
                    found[agent_key(a)] = a
                continue

            factory_matches = resolve_factory_agent(anchor, str(sym), records_by_file, by_file_symbol)
            if factory_matches:
                for a in factory_matches:
                    found[agent_key(a)] = a
                continue

            # Nested orchestration: if this exact symbol is itself the owner of
            # a composition object, recursively expand its members. This is
            # topology extraction, not framework-default inference.
            nested = composition_by_file[anchor].get(str(sym), [])
            nested_anchor = anchor
            nested_imported = imported
            if not nested:
                # The symbol may be a Team/Crew/etc. built in a different file
                # and imported into this one. Resolve it with the same
                # module-path exactness used for imported agents, then look
                # for its composition membership in that remote file.
                cand_files, attr = resolve_import_target(anchor, str(sym), imported, records_by_file)
                for cand in cand_files:
                    remote_nested = composition_by_file.get(cand, {}).get(attr, [])
                    if remote_nested:
                        nested = remote_nested
                        nested_anchor = cand
                        nested_imported = (records_by_file.get(cand, {}) or {}).get("imported_symbols", {}) or {}
                        cross_file_nested_composition_hits.add((anchor, str(sym)))
                        break

            if nested:
                nested_found, nested_unresolved = expand_members(
                    nested_anchor, nested, nested_imported, seen
                )
                found.update({agent_key(a): a for a in nested_found.values()})
                unresolved.extend(nested_unresolved)
                continue

            # A known orchestration symbol is not an unresolved reference;
            # it may be expanded by nested composition or intentionally remain
            # an orchestration-only object.
            if str(sym) in composition_by_file[anchor]:
                continue

            unresolved.append({
                "symbol": str(sym),
                "origin": imported.get(str(sym)),
                "reason": "composition_member_unresolved"
            })
        return found, unresolved

    systems = []
    anchor_files = sorted({e["file"] for e in edges})

    for anchor in anchor_files:
        local_edges = [e for e in edges if e["file"] == anchor]
        info = records_by_file.get(anchor, {})
        imported = info.get("imported_symbols", {}) or {}
        attached = {}
        unresolved = []

        # Explicit edge references.
        mentioned = set().union(*(edge_symbols(e) for e in local_edges))
        for sym in mentioned:
            found, miss = resolve_symbol(anchor, sym, imported)
            for a in found:
                attached[agent_key(a)] = a
            unresolved.extend(miss)

        # Composition member expansion, including nested Teams/Crews.
        for owner, members in composition_by_file[anchor].items():
            found, miss = expand_members(anchor, members, imported)
            for a in found.values():
                attached[agent_key(a)] = a
            unresolved.extend(miss)

        # A structural anchor establishes a bounded source-file system.
        # Include same-file agent instances when the source file itself is the
        # explicit example/application boundary. This is necessary for cases
        # such as OpenAI handoff examples where an agent is part of the example
        # but is not itself named in the handoff list.
        #
        # Orchestration units are already excluded from `agents`, so this does
        # not reintroduce SequentialAgent/Team/Crew as fake agents.
        for a in by_file.get(anchor, []):
            attached[agent_key(a)] = a

        if len(attached) < 2:
            continue

        files = sorted({anchor} | {a["file"] for a in attached.values()})
        cross_file = any(a["file"] != anchor for a in attached.values())

        systems.append({
            "system_id": f"{repo}::system::{len(systems)+1:04d}",
            "repository": repo,
            "files": files,
            "anchor_files": [anchor],
            "system_boundary_status": "explicit_structural_anchor",
            "discovery_evidence": {
                "anchor_evidence_types": sorted({e["evidence_type"] for e in local_edges}),
                "structural_anchor_count": len(local_edges),
                "import_linkage_used": cross_file,
                "nested_composition_evidence": (
                    any(e.get("evidence_type") == "nested_team_composition" for e in local_edges)
                    or any(a == anchor for a, _ in cross_file_nested_composition_hits)
                ),
                "unresolved_references": unresolved,
            },
            "agents": list(attached.values()),
            "interactions": local_edges,
        })

    # Merge only identical anchor identity. Separate source files remain
    # separate candidate systems even when they happen to use the same names.
    merged = []
    by_anchor = {}
    for s in systems:
        anchor = tuple(s["anchor_files"])
        if anchor not in by_anchor:
            by_anchor[anchor] = s
            merged.append(s)
        else:
            t = by_anchor[anchor]
            t["agents"] = canonical_agents(t["agents"] + s["agents"])
            t["interactions"] = dedupe_edges(t["interactions"] + s["interactions"])
            t["files"] = sorted(set(t["files"]) | set(s["files"]))
            t["discovery_evidence"]["anchor_evidence_types"] = sorted(
                set(t["discovery_evidence"]["anchor_evidence_types"]) |
                set(s["discovery_evidence"]["anchor_evidence_types"])
            )
            t["discovery_evidence"]["structural_anchor_count"] += s["discovery_evidence"]["structural_anchor_count"]
            t["discovery_evidence"]["import_linkage_used"] = (
                t["discovery_evidence"]["import_linkage_used"] or
                s["discovery_evidence"]["import_linkage_used"]
            )
            t["discovery_evidence"]["unresolved_references"].extend(
                s["discovery_evidence"].get("unresolved_references", [])
            )

    for i, s in enumerate(merged, 1):
        s["system_id"] = f"{repo}::system::{i:04d}"

    return merged

def finalize_system(s):
    agents = s["agents"]
    edges = s["interactions"]
    patterns = sorted({
        e["type"] for e in edges if e.get("type") in ARCHITECTURAL_TYPES
    })
    return {
        "system_id": s["system_id"],
        "repository": s["repository"],
        "files": s["files"],
        "anchor_files": s["anchor_files"],
        "system_boundary_status": s["system_boundary_status"],
        "discovery_evidence": s["discovery_evidence"],
        "agent_count": len(agents),
        "agents": agents,
        "interactions": edges,
        "responsibility_bearing_agents": sum(bool(a["responsibilities"]) for a in agents),
        "provenance": sorted({a["provenance"] for a in agents}),
        "architecture_primary": None,
        "architecture_patterns": patterns,
        "architecture_evidence_types": sorted({
            e.get("evidence_type") for e in edges if e.get("evidence_type")
        }),
        "interaction_primitives": sorted({
            x
            for e in edges
            for x in (
                [e.get("type")] +
                ({
                    "conditional_handoff": ["conditional_handoff"],
                    "hierarchical_delegation": ["hierarchical_delegation"],
                    "round_robin_group": ["round_robin_group"],
                    "sequential_task_dependency": ["sequential_task_dependency"],
                    "adk_parallel_worker": ["fan_out_fan_in"],
                    "human_in_the_loop": ["human_in_the_loop"],
                    "explicit_review_revision_loop": ["feedback_loop"],
                    "event_subscription": ["publish_subscribe", "feedback_loop"],
                    "metagpt_hire_sequence": ["sequential_task_dependency"],
                    # A team whose own members are themselves teams is, by
                    # definition, coordinating those sub-teams — this is a
                    # structural fact of the composition, not an assumed
                    # framework default.
                    "nested_team_composition": ["nested_team_composition", "hierarchical_coordination"],
                    "orchestrator_instruction_sequence": ["sequential_team_dependency", "sequential_dependency"],
                }.get(e.get("evidence_type"), []))
            )
            if x in INTERACTION_PRIMITIVES
        }),
        "architecture_status": "observed_structural_patterns" if patterns else (
            "composition_only" if any(e.get("type") == "composition" for e in edges)
            else "unresolved"
        ),
        "composition_evidence": sum(
            e.get("type") == "composition" or
            e.get("evidence_type") in {"orchestrator_membership", "agno_team_membership_only", "agno_team_mode_unknown"}
            for e in edges
        ),
        "explicit_interaction_edges": sum(
            bool(e.get("source_agent") and e.get("target_agent")) for e in edges
        ),
        "confidence": "HIGH" if any(e.get("confidence") == "HIGH" for e in edges) else "MEDIUM",
    }


def source_files(repo_dir: Path):
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(x.lower() in SKIP_DIRS for x in p.parts):
            continue
        suffix = p.suffix.lower()
        if suffix not in PY_EXTENSIONS | JS_EXTENSIONS | NOTEBOOK_EXTENSIONS:
            continue
        cap = MAX_NOTEBOOK_FILE_BYTES if suffix in NOTEBOOK_EXTENSIONS else MAX_FILE_BYTES
        try:
            if p.stat().st_size > cap:
                continue
        except OSError:
            continue
        yield p


def mine_repo(repo_dir: Path):
    repo = repo_name(repo_dir)
    records, raw_agents, raw_edges, all_fws = [], [], [], set()

    for p in source_files(repo_dir):
        rel = norm(p.relative_to(repo_dir))
        prov = provenance(rel, repo)
        rec = {
            "file": rel, "provenance": prov,
            "frameworks": [], "imported_symbols": {}, "factory_refs": {}
        }
        records.append(rec)

        if prov in {"test", "documentation", "framework_internal"}:
            continue

        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if p.suffix.lower() in PY_EXTENSIONS:
            a, e, fw, imp, factory_refs = extract_py(repo, rel, src)
        elif p.suffix.lower() in NOTEBOOK_EXTENSIONS:
            pseudo_source = notebook_code_source(src)
            if pseudo_source and len(pseudo_source.encode("utf-8", "ignore")) <= MAX_FILE_BYTES:
                a, e, fw, imp, factory_refs = extract_py(repo, rel, pseudo_source)
            else:
                a, e, fw, imp, factory_refs = [], [], set(), {}, {}
        else:
            a, e, fw, imp, factory_refs = extract_js(repo, rel, src)

        rec["frameworks"] = sorted(fw)
        rec["imported_symbols"] = imp
        rec["factory_refs"] = factory_refs
        raw_agents.extend(a)
        raw_edges.extend(e)
        all_fws.update(fw)

    raw_agents = canonical_agents(dedupe_agents(raw_agents))
    raw_edges = dedupe_edges(raw_edges)

    candidates = discover_systems(repo, records, raw_agents, raw_edges)
    systems = [finalize_system(s) for s in candidates]

    sys_agents = [a for s in systems for a in s["agents"]]
    sys_edges = [e for s in systems for e in s["interactions"]]

    return {
        "repository": repo,
        "frameworks": sorted(all_fws),
        "source_file_count": len(records),
        "systems": systems,
        "raw_agents_audit_only": raw_agents,
        "raw_interactions_audit_only": raw_edges,
        "audit": {
            "candidate_system_count": len(systems),
            "agents_in_systems": len(sys_agents),
            "responsibility_bearing_agents_in_systems": sum(bool(a["responsibilities"]) for a in sys_agents),
            "explicit_interaction_edges_in_systems": sum(
                bool(e.get("source_agent") and e.get("target_agent")) for e in sys_edges
            ),
            "architecture_evidence_in_systems": sorted({
                e["type"] for e in sys_edges if e.get("type") in ARCHITECTURAL_TYPES
            }),
            "raw_agent_count_audit_only": len(raw_agents),
            "raw_interaction_count_audit_only": len(raw_edges),
        }
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not REPOS_DIR.exists():
        print(f"[ERROR] Missing {REPOS_DIR}")
        return

    repo_dirs = sorted(
        p for p in REPOS_DIR.iterdir()
        if p.is_dir() and p.name not in SKIP_DIRS
    )

    results = []
    print("=" * 80)
    print("CODE MINER V6.6 — SOURCE-IDENTITY SYSTEM PIPELINE")
    print("=" * 80)
    print("Primary unit: candidate multi-agent system")
    print("Specialization: NOT inferred")
    print("Architecture primary: NOT inferred")
    print("Repository-wide raw detections: AUDIT ONLY")
    print()

    for i, rd in enumerate(repo_dirs, 1):
        print(f"[{i}/{len(repo_dirs)}] {rd.name}")
        try:
            r = mine_repo(rd)
            results.append(r)
            a = r["audit"]
            print(
                f"  files={r['source_file_count']} "
                f"systems={a['candidate_system_count']} "
                f"agents_in_systems={a['agents_in_systems']} "
                f"edges={a['explicit_interaction_edges_in_systems']} "
                f"raw_agents={a['raw_agent_count_audit_only']}"
            )
            for s in r["systems"]:
                print(
                    f"    SYSTEM {s['system_id']} | "
                    f"agents={s['agent_count']} | "
                    f"patterns={s['architecture_patterns']} | "
                    f"anchors={s['anchor_files']}"
                )
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    ep = DATA_DIR / "evidence_v6_6.json"
    ap = DATA_DIR / "audit_v6_6.txt"
    ep.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "CODE MINER V6.6 AUDIT",
        "=" * 80,
        f"Repositories processed: {len(results)}",
        "Primary denominator: agents attached to candidate systems.",
        "Raw repository-wide detections are diagnostic only.",
        "Specialization is NOT inferred.",
        "Architecture_primary is NOT inferred.",
        "",
    ]
    for r in results:
        a = r["audit"]
        lines += [
            r["repository"],
            f"  source_files={r['source_file_count']}",
            f"  candidate_systems={a['candidate_system_count']}",
            f"  agents_in_systems={a['agents_in_systems']}",
            f"  responsibility_bearing_agents={a['responsibility_bearing_agents_in_systems']}",
            f"  explicit_interaction_edges={a['explicit_interaction_edges_in_systems']}",
            f"  architecture_evidence={a['architecture_evidence_in_systems']}",
            f"  raw_agents_audit_only={a['raw_agent_count_audit_only']}",
            "",
        ]
        for s in r["systems"]:
            lines += [
                f"  SYSTEM {s['system_id']}",
                f"    boundary={s['system_boundary_status']}",
                f"    anchors={s['anchor_files']}",
                f"    files={s['files']}",
                f"    provenance={s['provenance']}",
                f"    agents={s['agent_count']}",
                f"    responsibilities={s['responsibility_bearing_agents']}",
                f"    architecture_patterns={s['architecture_patterns']}",
                f"    composition_evidence={s['composition_evidence']}",
                f"    explicit_edges={s['explicit_interaction_edges']}",
                f"    confidence={s['confidence']}",
                "",
            ]

    ap.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"[OK] {ep}")
    print(f"[OK] {ap}")


if __name__ == "__main__":
    main()