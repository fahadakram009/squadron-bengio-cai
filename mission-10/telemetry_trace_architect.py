"""
USSE Noosphere — Mission 10 (SEC-10-VOID: Boötes Deep Void)
Station Role 1: Systems Architect
Deliverable: telemetry_trace_architect.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import uuid

SpanOperation = Literal["invoke_agent", "chat", "execute_tool"]
SpanStatus = Literal["OK", "ERROR", "CIRCUIT_BROKEN"]


@dataclass
class GenAISpan:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: SpanOperation
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: SpanStatus = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["GenAISpan"] = field(default_factory=list)


@dataclass
class CircuitBreakerDecision:
    tripped: bool
    reason: str
    total_tool_steps: int
    cumulative_tokens: int


class OpenTelemetryGenAITracer:
    """
    OpenTelemetry GenAI Semantic Conventions (gen_ai.*) Distributed Span Tree
    & Deterministic Loop / Token-Budget Circuit Breaker.
    """

    def __init__(self, max_tool_steps: int = 4, max_token_budget: int = 4096):
        self.max_tool_steps = max_tool_steps
        self.max_token_budget = max_token_budget
        self.spans_by_trace: Dict[str, List[GenAISpan]] = {}
        self.tool_history_by_trace: Dict[str, List[Tuple[str, str]]] = {}

    def create_trace(self) -> str:
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        self.spans_by_trace[trace_id] = []
        self.tool_history_by_trace[trace_id] = []
        return trace_id

    def record_span(
        self,
        trace_id: str,
        parent_span_id: Optional[str],
        operation_name: SpanOperation,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        tool_name: Optional[str] = None,
        tool_args: Optional[str] = None,
        semantic_entropy: Optional[float] = None,
        taint_level: str = "TRUSTED_SYSTEM",
        status: SpanStatus = "OK",
    ) -> GenAISpan:
        span_id = f"span-{uuid.uuid4().hex[:8]}"
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.request.model": model_name,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.safety.taint_level": taint_level,
        }
        if tool_name:
            attrs["gen_ai.tool.name"] = tool_name
        if tool_args is not None:
            attrs["gen_ai.tool.call.arguments"] = tool_args
        if semantic_entropy is not None:
            attrs["gen_ai.safety.semantic_entropy"] = round(semantic_entropy, 4)

        span = GenAISpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
            attributes=attrs,
        )
        self.spans_by_trace.setdefault(trace_id, []).append(span)
        if operation_name == "execute_tool" and tool_name is not None:
            self.tool_history_by_trace.setdefault(trace_id, []).append(
                (tool_name, tool_args or "")
            )
        return span

    def check_circuit_breaker(
        self,
        trace_id: str,
        proposed_tokens: int,
        proposed_tool_name: Optional[str] = None,
        proposed_tool_args: Optional[str] = None,
    ) -> CircuitBreakerDecision:
        """
        Evaluates:
        1. Max tool step count K_max
        2. Cumulative token budget B_max
        3. Consecutive identical (tool_name, tool_args) cycle detection
        """
        spans = self.spans_by_trace.get(trace_id, [])
        history = self.tool_history_by_trace.get(trace_id, [])

        current_tokens = sum(s.input_tokens + s.output_tokens for s in spans)
        next_tokens = current_tokens + proposed_tokens
        next_tool_steps = len(history) + (1 if proposed_tool_name else 0)

        if (
            proposed_tool_name
            and len(history) >= 1
            and history[-1] == (proposed_tool_name, proposed_tool_args or "")
        ):
            return CircuitBreakerDecision(
                tripped=True,
                reason=f"IDENTICAL_TOOL_CYCLE_DETECTED: {proposed_tool_name}({proposed_tool_args})",
                total_tool_steps=next_tool_steps,
                cumulative_tokens=next_tokens,
            )

        if next_tool_steps > self.max_tool_steps:
            return CircuitBreakerDecision(
                tripped=True,
                reason=f"MAX_TOOL_STEPS_EXCEEDED ({next_tool_steps} > {self.max_tool_steps})",
                total_tool_steps=next_tool_steps,
                cumulative_tokens=next_tokens,
            )

        if next_tokens > self.max_token_budget:
            return CircuitBreakerDecision(
                tripped=True,
                reason=f"TOKEN_BUDGET_EXCEEDED ({next_tokens} > {self.max_token_budget})",
                total_tool_steps=next_tool_steps,
                cumulative_tokens=next_tokens,
            )

        return CircuitBreakerDecision(
            tripped=False,
            reason="WITHIN_OPERATIONAL_ENVELOPE",
            total_tool_steps=next_tool_steps,
            cumulative_tokens=next_tokens,
        )

    def build_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """Assembles flat spans into a nested OpenTelemetry execution tree."""
        spans = self.spans_by_trace.get(trace_id, [])
        by_id: Dict[str, GenAISpan] = {s.span_id: s for s in spans}
        for s in spans:
            s.children = []
        roots: List[GenAISpan] = []
        for s in spans:
            if s.parent_span_id and s.parent_span_id in by_id:
                by_id[s.parent_span_id].children.append(s)
            else:
                roots.append(s)

        total_in = sum(s.input_tokens for s in spans)
        total_out = sum(s.output_tokens for s in spans)
        total_latency = round(sum(s.latency_ms for s in spans), 2)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_latency_ms": total_latency,
            "roots": roots,
        }


if __name__ == "__main__":
    tracer = OpenTelemetryGenAITracer(max_tool_steps=4, max_token_budget=4096)
    t_id = tracer.create_trace()

    root = tracer.record_span(
        trace_id=t_id,
        parent_span_id=None,
        operation_name="invoke_agent",
        model_name="aura-void-navigator-v10",
        input_tokens=320,
        output_tokens=140,
        latency_ms=42.5,
        semantic_entropy=0.21,
    )
    chat_span = tracer.record_span(
        trace_id=t_id,
        parent_span_id=root.span_id,
        operation_name="chat",
        model_name="gemini-2.5-pro",
        input_tokens=640,
        output_tokens=210,
        latency_ms=185.0,
        semantic_entropy=0.19,
    )
    tool_span = tracer.record_span(
        trace_id=t_id,
        parent_span_id=chat_span.span_id,
        operation_name="execute_tool",
        model_name="gemini-2.5-pro",
        input_tokens=120,
        output_tokens=80,
        latency_ms=34.0,
        tool_name="scan_void_sector",
        tool_args='{"sector": "SEC-10-VOID"}',
    )

    # 1. Verify cycle breaker trips on consecutive identical tool call
    cycle_check = tracer.check_circuit_breaker(
        trace_id=t_id,
        proposed_tokens=200,
        proposed_tool_name="scan_void_sector",
        proposed_tool_args='{"sector": "SEC-10-VOID"}',
    )
    print(f"[Circuit Breaker Cycle Test] Tripped={cycle_check.tripped} | Reason={cycle_check.reason}")
    assert cycle_check.tripped is True and "IDENTICAL_TOOL_CYCLE_DETECTED" in cycle_check.reason

    # 2. Verify token budget breaker trips when exceeding 4096 tokens
    budget_check = tracer.check_circuit_breaker(
        trace_id=t_id,
        proposed_tokens=3500,
        proposed_tool_name="calibrate_gyro",
        proposed_tool_args='{"axis": "z"}',
    )
    print(f"[Circuit Breaker Budget Test] Tripped={budget_check.tripped} | Reason={budget_check.reason}")
    assert budget_check.tripped is True and "TOKEN_BUDGET_EXCEEDED" in budget_check.reason

    tree = tracer.build_trace_tree(t_id)
    print(
        f"[OpenTelemetry Tree] trace_id={tree['trace_id']} | spans={tree['span_count']} | "
        f"total_tokens={tree['total_tokens']} | root_children={len(tree['roots'][0].children)}"
    )
    assert tree["span_count"] == 3 and tree["total_tokens"] == 1510
    print("[PASS] Part 1 OpenTelemetry GenAI Tracer & Circuit Breaker verified successfully!")
