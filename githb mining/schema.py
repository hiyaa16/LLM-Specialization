from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class RepositoryRecord:
    repo_name: str
    repo_url: str
    description: str

    stars: int
    forks: int
    language: Optional[str]

    frameworks: List[str]

    # Multi-agent evidence
    agent_count_estimate: Optional[int]
    agent_names: List[str]
    responsibilities: Dict[str, str]

    # Architecture
    architecture_type: Optional[str]

    # Responsibility specialization
    specialization_level: Optional[str]
    specialization_ratio: Optional[float]

    # Interaction patterns
    planning: bool
    sequential: bool
    parallel: bool
    hierarchical: bool
    peer_to_peer: bool
    iterative: bool
    conditional_routing: bool
    shared_state: bool

    # Evidence
    evidence_files: List[str]
    evidence_snippets: List[str]

    confidence: Optional[float]

    def to_dict(self):
        return asdict(self)