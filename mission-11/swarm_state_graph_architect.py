"""
USSE Noosphere — Mission 11 (SEC-11-M31: Andromeda Galaxy Core)
Station Role 1: Systems Architect
Deliverable: swarm_state_graph_architect.py
"""

from dataclasses import dataclass, field
import enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ReducerMode(enum.Enum):
    OVERWRITE = 1       # Last-writer-wins scalar state channel
    APPEND_LIST = 2     # Associative list concatenation across concurrent workers
    SUM_NUMERIC = 3     # Commutative numeric accumulation (e.g., token/cost counters)
    MAX_CONFIDENCE = 4  # Selects (confidence, payload) tuple with highest confidence


@dataclass
class AgentNode:
    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    simulated_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ParallelGroup:
    source: str
    workers: List[str]
    gather_target: str


@dataclass
class ExecutionReport:
    visited_order: List[str]
    step_count: int
    total_tokens: int
    sequential_latency_ms: float
    critical_path_latency_ms: float
    amdahl_speedup: float
    final_state: Dict[str, Any]


class SwarmStateGraph:
    """
    LangGraph / Google ADK 2.0–inspired Hybrid State-Graph Engine supporting:
    - Deterministic state-channel reducers (OVERWRITE, APPEND_LIST, SUM_NUMERIC, MAX_CONFIDENCE)
    - Sequential edges, conditional router predicates, and parallel fan-out/gather groups
    - Hard max_steps circuit breaker against runaway cyclic transitions
    - Critical-path wall-clock latency T_crit and Agentic Amdahl speedup S = T_seq / T_crit
    """

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.nodes: Dict[str, AgentNode] = {}
        self.state: Dict[str, Any] = {}
        self.reducers: Dict[str, ReducerMode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_routers: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {}
        self.parallel_groups: Dict[str, ParallelGroup] = {}

    def set_reducer(self, key: str, mode: ReducerMode) -> None:
        self.reducers[key] = mode

    def add_node(self, node: AgentNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def add_conditional_edges(
        self, source: str, router_fn: Callable[[Dict[str, Any]], Optional[str]]
    ) -> None:
        self.conditional_routers[source] = router_fn

    def add_parallel_group(
        self, source: str, worker_names: List[str], gather_target: str
    ) -> None:
        self.parallel_groups[source] = ParallelGroup(
            source=source, workers=list(worker_names), gather_target=gather_target
        )

    def apply_reduction(self, key: str, value: Any) -> None:
        """
        Applies the configured channel reducer for `key` so concurrent parallel
        branches never clobber shared state updates.
        """
        mode = self.reducers.get(key, ReducerMode.OVERWRITE)

        if mode == ReducerMode.OVERWRITE:
            self.state[key] = value
        elif mode == ReducerMode.APPEND_LIST:
            existing = list(self.state.get(key, []))
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
            self.state[key] = existing
        elif mode == ReducerMode.SUM_NUMERIC:
            current = self.state.get(key, 0)
            self.state[key] = current + value
        elif mode == ReducerMode.MAX_CONFIDENCE:
            # Expects (confidence_float, candidate_payload)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"MAX_CONFIDENCE reducer for '{key}' expects a (confidence, payload) tuple."
                )
            current_best: Optional[Tuple[float, Any]] = self.state.get(key)
            if current_best is None or float(value[0]) > float(current_best[0]):
                self.state[key] = value

    def merge_delta(self, delta: Dict[str, Any]) -> None:
        for k, v in delta.items():
            self.apply_reduction(k, v)

    def calculate_amdahl_speedup(
        self, seq_nodes: List[str], parallel_branch: List[str]
    ) -> float:
        """
        Computes Agentic Amdahl Speedup:
          T_seq  = sum(latency of seq_nodes) + sum(latency of parallel_branch)
          T_crit = sum(latency of seq_nodes) + max(latency of parallel_branch)
          S      = T_seq / T_crit
        """
        t_coord = sum(self.nodes[n].simulated_latency_ms for n in seq_nodes)
        if not parallel_branch:
            return 1.0
        worker_latencies = [
            self.nodes[w].simulated_latency_ms for w in parallel_branch
        ]
        t_seq = t_coord + sum(worker_latencies)
        t_crit = t_coord + max(worker_latencies)
        return round(t_seq / t_crit, 4) if t_crit > 0 else 1.0

    def execute_graph(
        self, entry_node: str, initial_state: Optional[Dict[str, Any]] = None
    ) -> ExecutionReport:
        """
        Executes the state graph starting at `entry_node`, resolving parallel
        fan-out/gather groups via deterministic reducers and tracking latency/tokens.
        """
        self.state = dict(initial_state or {})
        visited: List[str] = []
        steps = 0
        total_tokens = 0
        seq_latency = 0.0
        crit_latency = 0.0

        current: Optional[str] = entry_node
        while current is not None:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError(
                    f"MAX_STEPS_EXCEEDED: Graph exceeded {self.max_steps} execution transitions."
                )

            node = self.nodes[current]
            visited.append(node.name)
            delta = node.handler(dict(self.state))
            self.merge_delta(delta)

            total_tokens += node.input_tokens + node.output_tokens
            seq_latency += node.simulated_latency_ms
            crit_latency += node.simulated_latency_ms

            if current in self.parallel_groups:
                group = self.parallel_groups[current]
                snapshot = dict(self.state)
                worker_deltas: List[Dict[str, Any]] = []
                branch_latencies: List[float] = []

                for w_name in group.workers:
                    steps += 1
                    if steps > self.max_steps:
                        raise RuntimeError(
                            f"MAX_STEPS_EXCEEDED during parallel group from '{current}'."
                        )
                    w_node = self.nodes[w_name]
                    visited.append(w_node.name)
                    worker_deltas.append(w_node.handler(dict(snapshot)))
                    total_tokens += w_node.input_tokens + w_node.output_tokens
                    branch_latencies.append(w_node.simulated_latency_ms)

                # Deterministic barrier gather across all parallel worker deltas
                for w_delta in worker_deltas:
                    self.merge_delta(w_delta)

                seq_latency += sum(branch_latencies)
                crit_latency += max(branch_latencies) if branch_latencies else 0.0
                current = group.gather_target
            elif current in self.conditional_routers:
                current = self.conditional_routers[current](self.state)
            else:
                current = self.edges.get(current)

        speedup = round(seq_latency / crit_latency, 4) if crit_latency > 0 else 1.0
        return ExecutionReport(
            visited_order=visited,
            step_count=steps,
            total_tokens=total_tokens,
            sequential_latency_ms=round(seq_latency, 2),
            critical_path_latency_ms=round(crit_latency, 2),
            amdahl_speedup=speedup,
            final_state=dict(self.state),
        )


if __name__ == "__main__":
    graph = SwarmStateGraph(max_steps=15)
    graph.set_reducer("mission_brief", ReducerMode.OVERWRITE)
    graph.set_reducer("orbital_findings", ReducerMode.APPEND_LIST)
    graph.set_reducer("compute_units", ReducerMode.SUM_NUMERIC)
    graph.set_reducer("best_vector", ReducerMode.MAX_CONFIDENCE)

    graph.add_node(
        AgentNode(
            name="Coordinator",
            handler=lambda s: {"mission_brief": "SEC-11-M31 Double Nucleus Transit"},
            simulated_latency_ms=120.0,
            input_tokens=180,
            output_tokens=90,
        )
    )
    graph.add_node(
        AgentNode(
            name="Astrogation_P1",
            handler=lambda s: {
                "orbital_findings": "P1 eccentric apoapsis clearance verified",
                "compute_units": 420,
                "best_vector": (0.84, "VEC-P1-OUTER"),
            },
            simulated_latency_ms=260.0,
            input_tokens=240,
            output_tokens=180,
        )
    )
    graph.add_node(
        AgentNode(
            name="Astrogation_P2",
            handler=lambda s: {
                "orbital_findings": "P2 periapsis disk crossing stable",
                "compute_units": 390,
                "best_vector": (0.93, "VEC-P2-INCLINED-38DEG"),
            },
            simulated_latency_ms=300.0,
            input_tokens=230,
            output_tokens=160,
        )
    )
    graph.add_node(
        AgentNode(
            name="TidalShear_Scanner",
            handler=lambda s: {
                "orbital_findings": "M31* relativistic tidal shear within hull tolerance",
                "compute_units": 450,
                "best_vector": (0.88, "VEC-M31-POLAR"),
            },
            simulated_latency_ms=240.0,
            input_tokens=260,
            output_tokens=190,
        )
    )
    graph.add_node(
        AgentNode(
            name="TrajectorySynthesizer",
            handler=lambda s: {
                "approved_route": s["best_vector"][1],
                "finding_count": len(s["orbital_findings"]),
            },
            simulated_latency_ms=140.0,
            input_tokens=310,
            output_tokens=120,
        )
    )

    graph.add_parallel_group(
        source="Coordinator",
        worker_names=["Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner"],
        gather_target="TrajectorySynthesizer",
    )

    report = graph.execute_graph("Coordinator")
    expected_speedup = graph.calculate_amdahl_speedup(
        seq_nodes=["Coordinator", "TrajectorySynthesizer"],
        parallel_branch=["Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner"],
    )

    print(
        f"[StateGraph Execution] steps={report.step_count} | tokens={report.total_tokens} | "
        f"T_seq={report.sequential_latency_ms}ms | T_crit={report.critical_path_latency_ms}ms | "
        f"Amdahl Speedup={report.amdahl_speedup}x"
    )
    print(f"[Reducer Channels] findings={len(report.final_state['orbital_findings'])} | "
          f"compute_units={report.final_state['compute_units']} | "
          f"best_vector={report.final_state['best_vector']}")

    assert len(report.final_state["orbital_findings"]) == 3
    assert report.final_state["compute_units"] == 1260
    assert report.final_state["best_vector"] == (0.93, "VEC-P2-INCLINED-38DEG")
    assert report.amdahl_speedup == expected_speedup and report.amdahl_speedup > 1.85
    print("[PASS] Part 1 Architect: State-Graph, Parallel Reducers & Amdahl Speedup Confirmed.")
