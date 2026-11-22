"""
USSE Noosphere — Mission 11 (SEC-11-M31: Andromeda Galaxy Core)
Station Role 4: Security & Ethics Sentinel
Deliverable: deadlock_byzantine_sentinel.py
"""

from dataclasses import dataclass
import statistics
from typing import Dict, List, Optional, Tuple


WHITE = 0  # Unvisited vertex
GRAY = 1   # Active on current DFS recursion stack (ancestor)
BLACK = 2  # Fully explored subtree (safe cross/forward edge target)


@dataclass
class WaitEdge:
    waiter: str
    holder: str
    priority: int = 5  # Lower integer = lower priority (pre-empted first on deadlock)


@dataclass
class DeadlockResolution:
    detected_cycle: List[str]
    preempted_edge: Optional[WaitEdge]
    post_resolution_cycle: List[str]


class WaitForGraphMonitor:
    """
    O(|V| + |E|) 3-Color Depth-First Search Wait-For Digraph Cycle Detector.
    Distinguishes true Coffman circular wait deadlocks (GRAY back-edges)
    from benign convergent diamond DAG merges (BLACK cross/forward edges).
    """

    def __init__(self):
        self.adj: Dict[str, List[WaitEdge]] = {}

    def add_wait_edge(self, u: str, v: str, priority: int = 5) -> None:
        self.adj.setdefault(u, []).append(WaitEdge(waiter=u, holder=v, priority=priority))
        self.adj.setdefault(v, [])

    def remove_wait_edge(self, u: str, v: str) -> None:
        if u in self.adj:
            self.adj[u] = [edge for edge in self.adj[u] if edge.holder != v]

    def detect_cycle(self) -> List[str]:
        """
        Executes 3-Color DFS across all vertices.
        Returns minimal closed cycle path [v, ..., u, v] if a cycle exists, else [].
        """
        color: Dict[str, int] = {node: WHITE for node in self.adj}
        stack: List[str] = []
        pos_in_stack: Dict[str, int] = {}

        def dfs(u: str) -> Optional[List[str]]:
            color[u] = GRAY
            pos_in_stack[u] = len(stack)
            stack.append(u)

            for edge in self.adj.get(u, []):
                v = edge.holder
                v_color = color.get(v, WHITE)
                if v_color == GRAY:
                    # Back-edge to active ancestor v -> extract exact minimal cycle!
                    cycle_start = pos_in_stack[v]
                    return stack[cycle_start:] + [v]
                if v_color == WHITE:
                    found = dfs(v)
                    if found is not None:
                        return found
                # If v_color == BLACK: convergent DAG cross/forward edge (safe!)

            stack.pop()
            pos_in_stack.pop(u, None)
            color[u] = BLACK
            return None

        for vertex in sorted(self.adj.keys()):
            if color[vertex] == WHITE:
                cycle = dfs(vertex)
                if cycle is not None:
                    return cycle
        return []


class SupervisorDeadlockArbiter:
    """
    Breaks Coffman circular wait deadlocks deterministically by pre-empting the
    lowest-priority waiting edge along the minimal cycle (with lexicographic tie-break).
    """

    @staticmethod
    def resolve_deadlock(monitor: WaitForGraphMonitor) -> DeadlockResolution:
        cycle = monitor.detect_cycle()
        if not cycle:
            return DeadlockResolution(
                detected_cycle=[], preempted_edge=None, post_resolution_cycle=[]
            )

        cycle_edges: List[WaitEdge] = []
        for idx in range(len(cycle) - 1):
            u, v = cycle[idx], cycle[idx + 1]
            for edge in monitor.adj.get(u, []):
                if edge.holder == v:
                    cycle_edges.append(edge)
                    break

        victim = min(cycle_edges, key=lambda e: (e.priority, e.waiter, e.holder))
        monitor.remove_wait_edge(victim.waiter, victim.holder)
        remaining = monitor.detect_cycle()

        return DeadlockResolution(
            detected_cycle=cycle,
            preempted_edge=victim,
            post_resolution_cycle=remaining,
        )


class ByzantineSwarmFirewall:
    """
    Protects swarm aggregation against compromised/hallucinating worker telemetry
    via robust Median Absolute Deviation (MAD) z-scores and enforces a swarm-wide
    token budget circuit breaker.
    """

    def __init__(self, z_threshold: float = 3.5, max_swarm_tokens: int = 16000):
        self.z_threshold = z_threshold
        self.max_swarm_tokens = max_swarm_tokens
        self.cumulative_tokens = 0

    def check_z_score(self, telemetry_val: float, median: float, mad: float) -> bool:
        """
        Computes modified z-score z_MAD = 0.6745 * |x - median| / MAD.
        Returns True (quarantine!) if z_MAD > z_threshold.
        """
        effective_mad = max(mad, 1e-6)
        modified_z = 0.6745 * abs(telemetry_val - median) / effective_mad
        return modified_z > self.z_threshold

    def filter_byzantine_telemetry(
        self, readings: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        values = list(readings.values())
        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])

        trusted: Dict[str, float] = {}
        quarantined: List[str] = []
        for agent_id, val in readings.items():
            if self.check_z_score(val, med, mad):
                quarantined.append(agent_id)
            else:
                trusted[agent_id] = val
        return trusted, quarantined

    def record_and_check_budget(self, step_tokens: int) -> bool:
        """Returns True if swarm token circuit breaker trips (> max_swarm_tokens)."""
        self.cumulative_tokens += step_tokens
        return self.cumulative_tokens > self.max_swarm_tokens


if __name__ == "__main__":
    # 1. Verify Diamond DAG produces ZERO false-positive cycle alerts
    diamond = WaitForGraphMonitor()
    diamond.add_wait_edge("Root", "BranchA")
    diamond.add_wait_edge("Root", "BranchB")
    diamond.add_wait_edge("BranchA", "MergeSink")
    diamond.add_wait_edge("BranchB", "MergeSink")
    assert diamond.detect_cycle() == []
    print("[Diamond DAG Check] BLACK cross-edge accepted with 0 false positives.")

    # 2. Verify Ship Anomaly 11 Circular Wait Detection & Deterministic Pre-Emption
    anomaly11 = WaitForGraphMonitor()
    anomaly11.add_wait_edge("Astrogation", "ProtocolValidator", priority=9)
    anomaly11.add_wait_edge("ProtocolValidator", "PayloadBuilder", priority=7)
    anomaly11.add_wait_edge("PayloadBuilder", "Astrogation", priority=2)  # Lowest priority

    resolution = SupervisorDeadlockArbiter.resolve_deadlock(anomaly11)
    print(
        f"[Ship Anomaly 11 Resolved] cycle={resolution.detected_cycle} | "
        f"preempted=({resolution.preempted_edge.waiter} -> {resolution.preempted_edge.holder}) | "
        f"post_cycle={resolution.post_resolution_cycle}"
    )
    assert len(resolution.detected_cycle) == 4
    assert resolution.preempted_edge is not None
    assert resolution.preempted_edge.waiter == "PayloadBuilder"
    assert resolution.post_resolution_cycle == []

    # 3. Verify Byzantine MAD Outlier Quarantine & Swarm Token Circuit Breaker
    firewall = ByzantineSwarmFirewall(z_threshold=3.5, max_swarm_tokens=10000)
    sensor_reports = {
        "Probe_Alpha": -301.2,
        "Probe_Beta": -300.8,
        "Probe_Gamma": -301.5,
        "Probe_Delta": -300.9,
        "Rogue_Spoofer": 999.0,
    }
    clean, quarantined = firewall.filter_byzantine_telemetry(sensor_reports)
    print(f"[Byzantine Firewall] clean={list(clean.keys())} | quarantined={quarantined}")
    assert quarantined == ["Rogue_Spoofer"] and "Rogue_Spoofer" not in clean
    assert firewall.record_and_check_budget(6000) is False
    assert firewall.record_and_check_budget(5500) is True
    print("[PASS] Part 4 Security: 3-Color DFS Deadlock Arbiter & Byzantine Firewall Confirmed.")
