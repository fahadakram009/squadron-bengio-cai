"""
USSE Noosphere — Mission 11 (SEC-11-M31: Andromeda Galaxy Core)
Station Role 2: Communications & Protocol Officer
Deliverable: handoff_contract_protocol.py
"""

from dataclasses import dataclass, field
import enum
import json
from typing import Any, Dict, List, Optional


class HandoffMode(enum.Enum):
    PEER_TRANSFER = 1    # ADK sub_agents transfer: active agent pointer switches to recipient
    SUBROUTINE_TOOL = 2  # ADK AgentTool call: caller stays on return stack until specialist returns


@dataclass
class HandoffEnvelope:
    envelope_id: str
    sender: str
    recipient: str
    mode: HandoffMode
    allowed_context_keys: List[str]
    payload: Dict[str, Any]
    required_output_fields: List[str]
    hop_count: int = 0
    max_hops: int = 5


@dataclass
class HandoffDispatchResult:
    envelope_id: str
    active_agent: str
    caller_stack: List[str]
    scoped_context: Dict[str, Any]
    compression_ratio: float
    token_reduction_pct: float


class ContextScopingGate:
    """
    Enforces Typed Inter-Agent Handoff Contracts & the Least-Context Principle:
    1. Rejects infinite delegation ping-pong whenever hop_count > max_hops.
    2. Projects bloated global_state down to whitelisted allowed_context_keys.
    3. Distinguishes SUBROUTINE_TOOL (caller pushed onto return stack) from
       PEER_TRANSFER (active agent pointer transferred directly to peer).
    4. Validates specialist output dicts against required_output_fields.
    """

    def __init__(self, initial_agent: str = "AndromedaCoordinator"):
        self.active_agent: str = initial_agent
        self.caller_stack: List[str] = []
        self.audit_log: List[HandoffDispatchResult] = []

    @staticmethod
    def _serialized_length(obj: Dict[str, Any]) -> int:
        return len(json.dumps(obj, sort_keys=True, default=str))

    def compute_compression_ratio(
        self, original: Dict[str, Any], compressed: Dict[str, Any]
    ) -> float:
        """
        Returns serialized byte/char ratio len(compressed) / len(original).
        Lower values indicate stronger Least-Context pruning (1 - ratio = reduction %).
        """
        orig_len = self._serialized_length(original)
        if orig_len == 0:
            return 0.0
        comp_len = self._serialized_length(compressed)
        return round(comp_len / float(orig_len), 4)

    def process_handoff(
        self, global_state: Dict[str, Any], envelope: HandoffEnvelope
    ) -> Dict[str, Any]:
        """
        Validates hop bounds, projects global_state onto allowed_context_keys,
        and updates active_agent / caller_stack according to HandoffMode.
        """
        if envelope.hop_count > envelope.max_hops:
            raise RuntimeError(
                f"HOP_LIMIT_EXCEEDED: Envelope '{envelope.envelope_id}' at hop "
                f"{envelope.hop_count} exceeds max_hops={envelope.max_hops}."
            )

        scoped_state: Dict[str, Any] = {
            k: global_state[k]
            for k in envelope.allowed_context_keys
            if k in global_state
        }
        scoped_state["_handoff_payload"] = dict(envelope.payload)

        if envelope.mode == HandoffMode.SUBROUTINE_TOOL:
            self.caller_stack.append(envelope.sender)
            self.active_agent = envelope.recipient
        elif envelope.mode == HandoffMode.PEER_TRANSFER:
            self.active_agent = envelope.recipient

        ratio = self.compute_compression_ratio(global_state, scoped_state)
        reduction_pct = round((1.0 - ratio) * 100.0, 2)

        self.audit_log.append(
            HandoffDispatchResult(
                envelope_id=envelope.envelope_id,
                active_agent=self.active_agent,
                caller_stack=list(self.caller_stack),
                scoped_context=scoped_state,
                compression_ratio=ratio,
                token_reduction_pct=reduction_pct,
            )
        )
        return scoped_state

    def complete_subroutine_return(
        self, envelope: HandoffEnvelope, specialist_output: Dict[str, Any]
    ) -> str:
        """
        Validates required_output_fields against the specialist contract and,
        for SUBROUTINE_TOOL calls, pops and restores the calling coordinator.
        """
        missing = [
            f for f in envelope.required_output_fields if f not in specialist_output
        ]
        if missing:
            raise ValueError(
                f"SCHEMA_CONTRACT_VIOLATION: Specialist '{envelope.recipient}' "
                f"omitted required output fields: {missing}"
            )

        if envelope.mode == HandoffMode.SUBROUTINE_TOOL and self.caller_stack:
            self.active_agent = self.caller_stack.pop()
        return self.active_agent


if __name__ == "__main__":
    gate = ContextScopingGate(initial_agent="AndromedaCoordinator")

    bloated_global_state = {
        "sector_id": "SEC-11-M31",
        "p1_ephemeris": {"ra": "00h42m44.3s", "dec": "+41d16m09s", "vel_kms": -301.0},
        "full_chat_history": ["turn_" + str(i) + "_" + ("x" * 120) for i in range(12)],
        "raw_sensor_dump": "NOISE_" * 200,
        "unrelated_crew_logs": ["galley inventory", "hydroponics cycle 44"],
    }

    # 1. Test SUBROUTINE_TOOL (AgentTool) Least-Context scoping & stack push/pop
    sub_env = HandoffEnvelope(
        envelope_id="ENV-SUB-01",
        sender="AndromedaCoordinator",
        recipient="AstrogationSpecialist",
        mode=HandoffMode.SUBROUTINE_TOOL,
        allowed_context_keys=["sector_id", "p1_ephemeris"],
        payload={"objective": "compute_p1_p2_clearance"},
        required_output_fields=["trajectory_id", "clearance_ly"],
        hop_count=1,
        max_hops=5,
    )
    scoped = gate.process_handoff(bloated_global_state, sub_env)
    last_audit = gate.audit_log[-1]
    print(
        f"[Subroutine Handoff] active={gate.active_agent} | stack={gate.caller_stack} | "
        f"compression_ratio={last_audit.compression_ratio} | "
        f"reduction={last_audit.token_reduction_pct}%"
    )
    assert "full_chat_history" not in scoped and "raw_sensor_dump" not in scoped
    assert gate.caller_stack == ["AndromedaCoordinator"]
    assert last_audit.token_reduction_pct > 75.0

    restored = gate.complete_subroutine_return(
        sub_env, {"trajectory_id": "M31-SAFE-07", "clearance_ly": 0.42}
    )
    assert restored == "AndromedaCoordinator" and gate.caller_stack == []

    # 2. Test PEER_TRANSFER active-agent switch without growing caller_stack
    peer_env = HandoffEnvelope(
        envelope_id="ENV-PEER-02",
        sender="AndromedaCoordinator",
        recipient="SafetyCritic",
        mode=HandoffMode.PEER_TRANSFER,
        allowed_context_keys=["sector_id"],
        payload={"trajectory_id": "M31-SAFE-07"},
        required_output_fields=["approved"],
        hop_count=2,
        max_hops=5,
    )
    gate.process_handoff(bloated_global_state, peer_env)
    assert gate.active_agent == "SafetyCritic" and gate.caller_stack == []

    # 3. Test HOP_LIMIT_EXCEEDED guard against infinite delegation loops
    runaway_env = HandoffEnvelope(
        envelope_id="ENV-LOOP-99",
        sender="SafetyCritic",
        recipient="AstrogationSpecialist",
        mode=HandoffMode.PEER_TRANSFER,
        allowed_context_keys=["sector_id"],
        payload={},
        required_output_fields=[],
        hop_count=6,
        max_hops=5,
    )
    try:
        gate.process_handoff(bloated_global_state, runaway_env)
        assert False, "Expected RuntimeError for HOP_LIMIT_EXCEEDED"
    except RuntimeError as exc:
        print(f"[Hop Guard Verified] Caught expected error: {exc}")
        assert "HOP_LIMIT_EXCEEDED" in str(exc)

    print("[PASS] Part 2 Protocol: Least-Context Scoping & Handoff Envelopes Confirmed.")
