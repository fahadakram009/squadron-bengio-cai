"""
Mission 08 — Part 2: Communications & Protocol Officer
3-Actor Reflexion Verbal RL & AST Traceback Surgeon (Shinn et al., NeurIPS 2023)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
"""

import hashlib
import re
import textwrap
from typing import Callable, Dict, List, Set, Tuple


class TracebackParser:
    """Helper utility to extract structured (file, line_number, exception_type, fault_summary) from Python tracebacks."""

    LINE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')
    EXC_PATTERN = re.compile(r"^([A-Za-z_]\w*(?:Error|Exception|Warning)):\s*(.*)$")
    # also matches a bare name ("AssertionError") and dotted names ("json.decoder.JSONDecodeError: ...")
    EXC_FALLBACK = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit|Iteration))(?::\s*(.*))?$")

    @classmethod
    def parse(cls, stderr_text: str) -> Dict[str, str]:
        file_name = "unknown.py"
        line_number = "-1"
        exception_type = "RuntimeError"
        fault_summary = stderr_text.strip()

        lines = [ln.strip() for ln in stderr_text.strip().splitlines() if ln.strip()]
        for ln in lines:
            m_loc = cls.LINE_PATTERN.search(ln)
            if m_loc:
                file_name, line_number = m_loc.group(1), m_loc.group(2)
            m_exc = cls.EXC_PATTERN.match(ln)
            if m_exc:
                exception_type, fault_summary = m_exc.group(1), m_exc.group(2)
            elif cls.EXC_FALLBACK.match(ln):
                m_alt = cls.EXC_FALLBACK.match(ln)
                exception_type, fault_summary = m_alt.group(1), m_alt.group(2) or "(no message)"

        return {
            "file": file_name,
            "line_number": line_number,
            "exception_type": exception_type,
            "fault_summary": fault_summary,
        }


class ReflexionSurgeon:
    """
    Implements the 3-Actor Verbal Reinforcement Learning loop (Shinn et al., 2023):
      - Actor (M_a): Generates candidate code patches conditioned on bounded episodic memory.
      - Evaluator (M_e): Executes deterministic unit tests / runtime verification (grounded external oracle).
      - Self-Reflection (M_sr): Translates traceback diagnostics into a natural-language critique r_k
        stored in a sliding episodic buffer capped at Omega = 3, with SHA-256 cycle detection.
    """

    def __init__(self, max_trials: int = 4, buffer_cap: int = 3):
        self.max_trials = max_trials
        self.buffer_cap = buffer_cap
        self.episodic_buffer: List[str] = []
        self.patch_hashes: Set[str] = set()

    def parse_traceback(self, stderr_text: str) -> Dict[str, str]:
        """
        Extract structured diagnostics from a Python traceback string.
        Return {'file': ..., 'line_number': ..., 'exception_type': ..., 'fault_summary': ...}.
        """
        return TracebackParser.parse(stderr_text)

    def generate_verbal_reflection(self, diag: Dict[str, str], failed_code: str) -> str:
        """
        Synthesize a structured natural-language reflection r_k diagnosing
        the root cause and prescribing a concrete code constraint for trial k+1.
        Append r_k to `self.episodic_buffer` while enforcing sliding-window capacity `self.buffer_cap`
        (evicting the oldest reflection when len(self.episodic_buffer) > self.buffer_cap).
        """
        exc = diag.get("exception_type", "Exception")
        line_no = diag.get("line_number", "?")
        summary = diag.get("fault_summary", "")

        if exc == "ZeroDivisionError":
            prescription = "Guard denominator against zero before dividing telemetry flux."
        elif exc == "KeyError":
            prescription = f"Use dict.get() or verify key {summary} exists in telemetry packet."
        elif exc == "IndexError":
            prescription = "Check array bounds before indexing shockwave sample buffer."
        elif exc == "AssertionError":
            prescription = "Unit test expectation not met: compare the returned value with the expected one."
        else:
            prescription = f"Resolve {exc} ({summary}) with explicit input validation."

        reflection = (
            f"[Reflection Line {line_no} | {exc}]: Fault '{summary}'. "
            f"Constraint for next trial: {prescription}"
        )
        code_lines = failed_code.splitlines()
        if line_no.isdigit() and 1 <= int(line_no) <= len(code_lines):  # point at the failing line
            reflection += f" Faulty line: `{code_lines[int(line_no) - 1].strip()}`."
        self.episodic_buffer.append(reflection)
        if len(self.episodic_buffer) > self.buffer_cap:
            self.episodic_buffer.pop(0)
        return reflection

    def is_duplicate_patch(self, candidate_code: str) -> bool:
        """
        Compute SHA-256 hash of whitespace-normalized `candidate_code`.
        Return True if already seen in `self.patch_hashes`, else record the digest and return False.
        """
        norm = "\n".join(line.strip() for line in candidate_code.strip().splitlines() if line.strip())
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if digest in self.patch_hashes:
            return True
        self.patch_hashes.add(digest)
        return False

    def run_repair_loop(
        self,
        initial_code: str,
        evaluator_fn: Callable[[str], Tuple[bool, str]],
        actor_patch_fn: Callable[[str, List[str]], str],
    ) -> Dict[str, object]:
        """
        Execute up to `self.max_trials` iterations of:
          1. Check if `initial_code` / `current_code` is a duplicate patch via `is_duplicate_patch`.
          2. Evaluate `current_code` via `evaluator_fn(current_code) -> (passed, stderr)`.
          3. If `passed` is True, return a success report dictionary.
          4. Otherwise, parse the traceback (`parse_traceback`), generate a verbal reflection
             (`generate_verbal_reflection`), and synthesize the next candidate patch via
             `actor_patch_fn(current_code, list(self.episodic_buffer))`.
        """
        current_code = initial_code
        history: List[Dict[str, object]] = []
        seen_layouts: Set[str] = set()

        for trial in range(1, self.max_trials + 1):
            duplicate = self.is_duplicate_patch(current_code)
            layout = self._layout_digest(current_code)
            if duplicate and layout in seen_layouts:
                # the Actor repeated a patch from this run: the deterministic tests would fail
                # the same way again, so stop instead of spending the remaining trials
                history.append({"trial": trial, "passed": False, "duplicate": True,
                                "reflection": "CYCLE: duplicate patch, repair loop stopped"})
                return {
                    "success": False,
                    "trials_used": trial,
                    "final_code": current_code,
                    "episodic_buffer": list(self.episodic_buffer),
                    "history": history,
                    "stopped_on_cycle": True,
                }
            seen_layouts.add(layout)
            passed, stderr = evaluator_fn(current_code)

            if passed:
                history.append({"trial": trial, "passed": True, "duplicate": duplicate, "reflection": "PASS"})
                return {
                    "success": True,
                    "trials_used": trial,
                    "final_code": current_code,
                    "episodic_buffer": list(self.episodic_buffer),
                    "history": history,
                }

            diag = self.parse_traceback(stderr)
            reflection = self.generate_verbal_reflection(diag, current_code)
            history.append(
                {
                    "trial": trial,
                    "passed": False,
                    "duplicate": duplicate,
                    "exception": diag["exception_type"],
                    "reflection": reflection,
                }
            )

            if trial < self.max_trials:
                next_code = actor_patch_fn(current_code, list(self.episodic_buffer))
                current_code = next_code

        return {
            "success": False,
            "trials_used": self.max_trials,
            "final_code": current_code,
            "episodic_buffer": list(self.episodic_buffer),
            "history": history,
        }

    @staticmethod
    def _layout_digest(code: str) -> str:
        """Like the duplicate hash, but keeps relative indentation (it changes what Python code does)."""
        lines = [ln.rstrip() for ln in textwrap.dedent(code).splitlines() if ln.strip()]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --- extra checks (Madhu) -------------------------------------------------------------

DEMO_V1 = """
def normalize_xray_flux(packet):
    return packet['flux'] / packet['window_sec']
"""
DEMO_V2 = """
def normalize_xray_flux(packet):
    if packet['window_sec'] == 0:
        return 0.0
    return packet['flux'] / packet['window_sec']
"""
DEMO_V3 = """
def normalize_xray_flux(packet):
    flux = packet.get('flux', 0.0)
    window = packet.get('window_sec', 1.0)
    if window == 0:
        return 0.0
    return flux / window
"""


def run_real_tests(code_str: str) -> Tuple[bool, str]:
    """Evaluator M_e: really runs the patch against unit tests and returns the real traceback."""
    import traceback
    scope: Dict[str, object] = {}
    try:
        exec(compile(code_str, "telemetry_parser.py", "exec"), scope)
        f = scope["normalize_xray_flux"]
        assert f({"flux": 10.0, "window_sec": 2.0}) == 5.0
        assert f({"flux": 10.0, "window_sec": 0}) == 0.0
        assert f({"window_sec": 4.0}) == 0.0
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def demo_actor(curr_code: str, reflections: List[str]) -> str:
    """Actor M_a: picks the next patch from the latest reflection (a real system would ask the LLM)."""
    latest = reflections[-1] if reflections else ""
    if "ZeroDivisionError" in latest:
        return DEMO_V2
    if "KeyError" in latest:
        return DEMO_V3
    return curr_code


def print_trajectory() -> None:
    report = ReflexionSurgeon(max_trials=4, buffer_cap=3).run_repair_loop(DEMO_V1, run_real_tests, demo_actor)
    print("Repair trajectory with real unit tests (doc Section B):")
    for step in report["history"]:
        exc = step.get("exception", "None")
        print(f"  Trial {step['trial']}: exception={exc} | duplicate={step['duplicate']} | "
              f"{'PASS' if step['passed'] else 'FAIL'} | {step['reflection']}")
    print(f"  success={report['success']} in {report['trials_used']} trials\n")


def run_tests() -> int:
    tests = []

    tb = ('Traceback (most recent call last):\n  File "telemetry_parser.py", line 3, in f\n'
          'ZeroDivisionError: division by zero')
    tests.append(("parse: file, line, exception and message",
                  ReflexionSurgeon().parse_traceback(tb) == {
                      "file": "telemetry_parser.py", "line_number": "3",
                      "exception_type": "ZeroDivisionError", "fault_summary": "division by zero"}))
    bare = ReflexionSurgeon().parse_traceback('  File "test_x.py", line 7, in test\nAssertionError')
    tests.append(("parse: bare AssertionError (no message)",
                  bare["exception_type"] == "AssertionError" and bare["fault_summary"] == "(no message)"
                  and bare["line_number"] == "7"))
    dotted = ReflexionSurgeon().parse_traceback(
        '  File "io.py", line 2, in load\njson.decoder.JSONDecodeError: Expecting value')
    tests.append(("parse: dotted exception name", dotted["exception_type"] == "json.decoder.JSONDecodeError"))
    chained = ReflexionSurgeon().parse_traceback(
        '  File "a.py", line 1, in f\nKeyError: \'x\'\n\nDuring handling of the above exception, '
        'another exception occurred:\n\n  File "a.py", line 4, in f\nValueError: bad packet')
    tests.append(("parse: chained traceback keeps the last exception and line",
                  chained["exception_type"] == "ValueError" and chained["line_number"] == "4"))

    s = ReflexionSurgeon(buffer_cap=3)
    for i in range(1, 6):
        s.generate_verbal_reflection({"exception_type": "ValueError", "line_number": str(i),
                                      "fault_summary": f"f{i}"}, "x = 1")
    tests.append(("episodic buffer = Tail_3: 5 reflections in, last 3 kept",
                  len(s.episodic_buffer) == 3 and "f3" in s.episodic_buffer[0] and "f5" in s.episodic_buffer[-1]))
    r = ReflexionSurgeon().generate_verbal_reflection(
        {"exception_type": "ZeroDivisionError", "line_number": "3", "fault_summary": "division by zero"}, DEMO_V1)
    tests.append(("reflection prescribes a constraint and quotes the faulty line",
                  "Guard denominator" in r and "packet['flux'] / packet['window_sec']" in r))

    s = ReflexionSurgeon()
    first = s.is_duplicate_patch(DEMO_V2)
    tests.append(("duplicate hash ignores trailing spaces and blank lines",
                  first is False and s.is_duplicate_patch(DEMO_V2.replace("\n", "  \n\n")) is True))

    calls = []
    stuck = ReflexionSurgeon(max_trials=4).run_repair_loop(
        DEMO_V1, lambda c: (calls.append(c) or run_real_tests(c)), lambda c, refl: c)
    tests.append(("cycle guard: Actor repeats its patch -> stopped at trial 2, tests run once",
                  stuck["success"] is False and stuck.get("stopped_on_cycle") is True
                  and stuck["trials_used"] == 2 and len(calls) == 1))

    a = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    b = "def f(x):\n    if x:\n        return 1\n        return 2\n"
    seq = iter([b, b])
    moved = ReflexionSurgeon(max_trials=3).run_repair_loop(
        a, lambda c: (False, '  File "f.py", line 4, in f\nAssertionError'), lambda c, refl: next(seq))
    tests.append(("indentation-only change is a new patch, not a cycle",
                  moved["history"][1].get("reflection", "").startswith("CYCLE") is False
                  and moved.get("stopped_on_cycle") is True and moved["trials_used"] == 3))

    never = ReflexionSurgeon(max_trials=3).run_repair_loop(
        "x = 1", lambda c: (False, "ValueError: still broken"), lambda c, refl: c + "\n# try again")
    tests.append(("budget: never passing -> stops after max_trials",
                  never["success"] is False and never["trials_used"] == 3 and len(never["history"]) == 3))

    real = ReflexionSurgeon(max_trials=4, buffer_cap=3).run_repair_loop(DEMO_V1, run_real_tests, demo_actor)
    h = real["history"]
    tests.append(("real unit tests: ZeroDivisionError -> KeyError -> PASS in 3 trials, no duplicates",
                  real["success"] is True and real["trials_used"] == 3
                  and [x.get("exception") for x in h[:2]] == ["ZeroDivisionError", "KeyError"]
                  and not any(x["duplicate"] for x in h)))

    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = sum(not ok for _, ok in tests)
    print(f"--- {len(tests) - failed} passed, {failed} failed ---")
    return failed


# ==============================================================================
# Verification Harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 2): Reflexion Surgeon & Traceback Parser Verification ===")

    surgeon = ReflexionSurgeon(max_trials=4, buffer_cap=3)

    BUGGY_CODE_V1 = """
def normalize_xray_flux(packet):
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V2 = """
def normalize_xray_flux(packet):
    if packet['window_sec'] == 0:
        return 0.0
    return packet['flux'] / packet['window_sec']
"""

    PATCH_V3 = """
def normalize_xray_flux(packet):
    flux = packet.get('flux', 0.0)
    window = packet.get('window_sec', 1.0)
    if window == 0:
        return 0.0
    return flux / window
"""

    def mock_unit_test_evaluator(code_str: str) -> Tuple[bool, str]:
        if "packet.get('flux'" not in code_str and "if packet['window_sec'] == 0" not in code_str:
            return (
                False,
                'Traceback (most recent call last):\n'
                '  File "telemetry_parser.py", line 3, in normalize_xray_flux\n'
                'ZeroDivisionError: division by zero when window_sec=0',
            )
        if "packet.get('flux'" not in code_str:
            return (
                False,
                'Traceback (most recent call last):\n'
                '  File "telemetry_parser.py", line 5, in normalize_xray_flux\n'
                "KeyError: 'flux'",
            )
        return True, ""

    def mock_actor(curr_code: str, reflections: List[str]) -> str:
        latest = reflections[-1] if reflections else ""
        if "ZeroDivisionError" in latest:
            return PATCH_V2
        if "KeyError" in latest:
            return PATCH_V3
        return curr_code

    report = surgeon.run_repair_loop(BUGGY_CODE_V1, mock_unit_test_evaluator, mock_actor)
    print(f"Repair Success: {report['success']} in {report['trials_used']} trials")
    for step in report["history"]:  # type: ignore[union-attr]
        print(f"  Trial {step['trial']}: passed={step['passed']} | {step['reflection']}")

    assert report["success"] is True and report["trials_used"] == 3
    assert surgeon.is_duplicate_patch(PATCH_V3) is True, "Duplicate patch SHA-256 guard failed!"
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")

    print()
    print_trajectory()
    if run_tests():
        raise SystemExit(1)
