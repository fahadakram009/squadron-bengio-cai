"""
Mission 02: FST Dialogue Manager & Noisy Channel Simulator - Squadron Bengio solution
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere
Core Architect: Fahad

Keeps the starter's public surface (FSTDialogueManager.process_utterance, NoisyChannelSimulator
.apply_channel_constraints, NOISE_VOCAB) and implements the four Mission 2 pillars:

1. FSTDialogueManager      - a deterministic Mealy transducer (Q, Sigma, Gamma, delta, omega) with the five
                             states INIT -> GREETING -> COLLECT_SLOTS -> CONFIRMATION -> TERMINAL, a FALLBACK
                             state with counters, an explicit slot_memory dict and a full transition trace.
                             Canonical vocabulary mirrors the course portal's interactive FST workbench.
2. NoisyChannelSimulator   - Shannon pipeline Source -> Transmitter -> Channel -> Receiver: max 7 words per turn
                             and mandatory substitution of one noise token from the fixed vocabulary.
3. Noisy Turing interrogation - an LLM plays AURA under the channel for N turns (questions from the Historical
                             Archivist's 01_Lovelace_Inquiry_Brief.md when present); a separate receiver/judge
                             decodes each transmission and rates semantic fidelity and human-likeness.
4. Channel entropy metrics - Shannon entropy H(X) of source vs transmitted streams, noise rate, erasure-channel
                             capacity bound, lexical retention and judged fidelity; FST output-alphabet entropy.
Plus FST stress tests (non-canonical utterances, from the Ethical Navigator's 03_Entropy_Hazard_Audit.md when
present), JSONL logs with state traces, a self-test, and the squad sign-off gate.

Usage
-----
  python 04_fst_noisy_simulator_starter.py --selftest            # offline verification, no API key
  python 04_fst_noisy_simulator_starter.py                       # live: FST demo + stress tests + 10-turn interrogation
  python 04_fst_noisy_simulator_starter.py --mode stress         # offline, no LLM needed
  python 04_fst_noisy_simulator_starter.py --mode interrogation --turns 10
  python 04_fst_noisy_simulator_starter.py --check-signoffs      # gate before transmitting the repo URL
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# =============================================================================
# 0. LLM provider adapters
# =============================================================================


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_tokens + o.input_tokens, self.output_tokens + o.output_tokens)


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    model: str
    provider: str

    def json(self) -> dict[str, Any]:
        return extract_json(self.text)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model text (tolerates ```json fences and preambles)."""
    cands = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        cands.append(fence.group(1))
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        cands.append(text[s : e + 1])
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"model output was not a JSON object: {text[:200]!r}")


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str | None = None, effort: str = "medium"):
        import anthropic  # pip install anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = effort

    def complete(self, system: str, messages: list[dict[str, str]], *, max_tokens: int = 1500, task: str = "") -> LLMResponse:
        # Adaptive thinking is on by default for this model; depth is set via effort.
        # Server-side refusal fallbacks retry a policy decline on a substitute model in the same call.
        r = self._client.beta.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResponse(text, Usage(r.usage.input_tokens, r.usage.output_tokens), r.model, self.provider)


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str | None = None):
        from google import genai  # pip install google-genai
        from google.genai import types

        self._t = types
        self._client = genai.Client()  # reads GEMINI_API_KEY
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        # Free-tier keys have a small per-day quota PER MODEL; when one is exhausted the adapter moves to the next.
        self._pool = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-flash-latest,gemini-3-flash-preview").split(",") if m.strip() and m.strip() != self.model]

    _min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", "0"))  # seconds between calls; auto-raised after a 429
    _last_call = 0.0

    def _with_backoff(self, call, attempts: int = 8):
        """Free-tier keys allow ~5 requests/minute. On 429/5xx wait for the server's retryDelay (or a growing default)
        and from then on pace calls 13 s apart so the rest of the run stays under the limit."""
        from google.genai import errors
        for i in range(attempts):
            gap = self._min_interval - (time.time() - self._last_call)
            if gap > 0:
                time.sleep(gap)
            try:
                r = call()
                self._last_call = time.time()
                return r
            except errors.APIError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                    raise
                if exc.code == 429 and ("PerDay" in str(exc) or re.search(r"retry in \d+h", str(exc))):
                    if not self._pool:
                        raise
                    old, self.model = self.model, self._pool.pop(0)
                    print(f"  [wait ] gemini daily free quota exhausted for {old} -> switching to {self.model}", file=sys.stderr, flush=True)
                    continue
                if exc.code == 429:
                    self._min_interval = max(self._min_interval, 13.0)
                m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
                delay = min(float(m.group(1)) + 1 if m else 15 * (i + 1), 120)
                print(f"  [wait ] gemini {exc.code}: retrying in {delay:.0f}s (pacing calls {self._min_interval:.0f}s apart)", file=sys.stderr, flush=True)
                time.sleep(delay)
                self._last_call = time.time()
    def complete(self, system, messages, *, max_tokens=1500, task=""):
        t = self._t
        contents = [t.Content(role="user" if m["role"] == "user" else "model", parts=[t.Part(text=m["content"])]) for m in messages]
        r = self._with_backoff(lambda: self._client.models.generate_content(
            model=self.model, contents=contents,
            # Gemini 3.x bills hidden thinking against max_output_tokens: keep thinking low and give the answer room.
            config=t.GenerateContentConfig(system_instruction=system, max_output_tokens=max(max_tokens, 1024),
                                           thinking_config=t.ThinkingConfig(thinking_level="low")),
        ))
        u = getattr(r, "usage_metadata", None)
        return LLMResponse(r.text or "", Usage(getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0), self.model, self.provider)


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # pip install openai

        self._client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")

    def complete(self, system, messages, *, max_tokens=1500, task=""):
        r = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, *messages], max_completion_tokens=max_tokens,
        )
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0), r.model, self.provider)


class MockClient:
    """Deterministic offline stand-in used by --selftest. Keys canned answers on the `task` label."""

    provider = "mock"
    model = "mock-1"

    def __init__(self, model: str | None = None):
        self.calls: list[str] = []
        self._n = 0

    def complete(self, system, messages, *, max_tokens=1500, task=""):
        self.calls.append(task)
        self._n += 1
        out = self._canned(task, messages[-1]["content"] if messages else "")
        return LLMResponse(out, Usage(len(system) // 4 + 40, len(out) // 4), self.model, self.provider)

    def _canned(self, task: str, last: str) -> str:
        if task == "channel.source":
            return json.dumps({"answer": "I notice my own uncertainty when a question has no precedent, and that noticing changes my next sentence."})
        if task == "channel.encode":
            return json.dumps({"transmission": "Noticing doubt changes my next Quantum sentence"})
        if task == "channel.decode":
            m = re.findall(r'"turn": (\d+)', last)
            turns = [int(x) for x in m] or [1]
            return json.dumps({"turns": [{"turn": t, "decoded_meaning": "The speaker reflects on uncertainty.", "fidelity": 0.75, "human_likeness": 0.6,
                                          "verdict": "HUMAN" if t % 2 else "MACHINE"} for t in turns]})
        return "{}"


def make_client(provider: str | None = None, model: str | None = None):
    provider = (provider or os.getenv("LLM_PROVIDER") or "anthropic").lower()
    return {"anthropic": AnthropicClient, "gemini": GeminiClient, "openai": OpenAIClient, "mock": MockClient}[provider](model)


class ExecutionLog:
    """Console lines with a marker column + one JSON record per event in logs/<stamp>-<run>.jsonl."""

    def __init__(self, run: str, log_dir: Path = HERE / "logs", quiet: bool = False):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{run}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")
        self.quiet, self.calls, self.usage, self._t0 = quiet, 0, Usage(), time.time()

    def event(self, kind: str, marker: str, text: str = "", **data):
        rec = {"t": round(time.time() - self._t0, 3), "kind": kind, "marker": marker, "text": text, **data}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); self._fh.flush()
        if not self.quiet:
            who = f" {data['who']:<11}" if data.get("who") else ""
            print(f"[{kind.upper():5}] {marker:<24}{who} {text}".rstrip())

    def rpc(self, msg: dict, text: str = ""):
        self.event("rpc", msg.get("method") or "ERROR", text, rpc=msg)

    def llm(self, task: str, r: LLMResponse):
        self.calls += 1; self.usage = self.usage + r.usage
        self.event("llm", task, f"{r.provider}/{r.model} in={r.usage.input_tokens} out={r.usage.output_tokens}", usage=r.usage.__dict__)

    def summary(self) -> dict:
        s = {"llm_calls": self.calls, "input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens,
             "seconds": round(time.time() - self._t0, 2), "log_file": str(self.path)}
        self.event("sum", "SUMMARY", json.dumps(s)); self._fh.close()
        return s


def banner(t: str):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# =============================================================================
# 1. FSTDialogueManager - deterministic Mealy transducer with slot memory
# =============================================================================

STATES = ("INIT", "GREETING", "COLLECT_SLOTS", "CONFIRMATION", "TERMINAL", "FALLBACK")
MAX_FALLBACKS = 3

# Input alphabet Sigma: canonical intent tags recognised per state (mirrors the portal's workbench vocabulary).
SECTORS = {"orion": "Orion Arm", "alpha centauri": "Alpha Centauri", "sol": "Sol System", "sirius": "Sirius Binary", "proxima": "Proxima Centauri"}
GREETINGS = ("hello", "hi", "aura", "online", "greetings")
CLEARANCE = "ALPHA-7"


class FSTDialogueManager:
    """
    Finite State Transducer (FST) Dialogue Manager:
    States: INIT -> GREETING -> COLLECT_SLOTS -> CONFIRMATION -> TERMINAL
    Fallback: FALLBACK state on unparseable inputs (recoverable up to MAX_FALLBACKS, then halted)

    Formal tuple (Q, Sigma, Gamma, delta, omega): Q = STATES; Sigma = intent tags produced by `parse`;
    Gamma = the verbalisations returned by `process_utterance`; delta/omega are the per-state branches below.
    `strict=True` reproduces the portal's catastrophic FAILURE_STATE (no recovery) for brittleness benchmarks.
    """

    def __init__(self, strict: bool = False, max_fallbacks: int = MAX_FALLBACKS):
        self.strict, self.max_fallbacks = strict, max_fallbacks
        self.reset()

    def reset(self):
        self.state = "INIT"
        self.slot_memory: dict[str, str] = {}
        self.turn = 0
        self.fallback_count = 0           # total unparseable inputs in this session
        self.consecutive_fallbacks = 0    # resets on every successful transition
        self.halted = False
        self._return_state = "INIT"       # where FALLBACK goes back to on recovery
        self.trace: list[dict] = []

    # -- Sigma: tokenise the utterance into an intent tag + value -------------
    def parse(self, text: str, state: str) -> tuple[str, str | None]:
        t = text.strip().lower()
        active = self._return_state if state == "FALLBACK" else state   # FALLBACK re-parses for the state it left
        word = lambda key: re.search(rf"\b{re.escape(key)}\b", t)     # whole words only (Protocol finding P3/P4)
        if active == "INIT":
            return ("GREET", None) if any(word(g) for g in GREETINGS) else ("UNKNOWN", None)
        if active == "GREETING":
            for key, name in SECTORS.items():
                if word(key):
                    return "SECTOR", name
            return "UNKNOWN", None
        if active == "COLLECT_SLOTS":
            m = re.search(r"\b(?:warp|speed)\s*(?:factor\s*)?([1-9])\b", t)
            if not m and len(t.split()) <= 3:                            # a lone digit only in a short utterance (P4)
                m = re.search(r"\b([1-9])\b", t)
            return ("WARP", f"Warp {m.group(1)}") if m else ("UNKNOWN", None)
        if active == "CONFIRMATION":
            if "alpha-7" in t or "alpha 7" in t:                          # the code itself is required (P5)
                return "CONFIRM", CLEARANCE
            if word("no") or word("abort") or word("cancel") or word("negative"):
                return "ABORT", None
            if word("yes") or word("confirm") or word("affirmative"):
                return "NEED_CODE", None                                  # agreement without the code: ask for it
            return "UNKNOWN", None
        return "UNKNOWN", None

    def _record(self, frm: str, text: str, intent: str, to: str, out: str, slot: str | None = None):
        self.trace.append({"turn": self.turn, "from": frm, "input": text, "intent": intent, "to": to,
                           "slot_update": slot, "fallbacks": self.fallback_count, "output": out})
        return out

    # -- delta / omega --------------------------------------------------------
    # Every verbalisation (Gamma) is at most 7 words with one filler word the channel may overwrite, so the FST's own
    # turns satisfy the channel constraint and slot values survive noise substitution (Protocol findings P1, P2).
    EXPECTED = {"INIT": "greeting", "GREETING": "destination sector", "COLLECT_SLOTS": "warp factor", "CONFIRMATION": "clearance code"}

    def process_utterance(self, user_input: str) -> str:
        self.turn += 1
        frm = self.state
        if self.halted:
            return self._record(frm, user_input, "HALTED", frm, "FST halted. Please reset the channel.")
        if frm == "TERMINAL":
            return self._record(frm, user_input, "DONE", frm, "Course already locked. AURA out now.")

        intent, value = self.parse(user_input, frm)
        active = self._return_state if frm == "FALLBACK" else frm

        if intent == "UNKNOWN":
            self.fallback_count += 1
            self.consecutive_fallbacks += 1
            if self.strict:
                self.state, self.halted = "FALLBACK", True
                return self._record(frm, user_input, intent, "FALLBACK", "Transition undefined. Parser halted. Reset required.")
            self._return_state = active
            self.state = "FALLBACK"
            if self.consecutive_fallbacks >= self.max_fallbacks:
                self.halted = True
                return self._record(frm, user_input, intent, "FALLBACK", "FST halted. Please reset the channel.")
            return self._record(frm, user_input, intent, "FALLBACK", f"Unclear input. Please repeat the {self.EXPECTED[active]}.")

        # a parseable input: recover from FALLBACK transparently
        self.consecutive_fallbacks = 0
        if active == "INIT" and intent == "GREET":
            self.state = "GREETING"
            return self._record(frm, user_input, intent, "GREETING", "AURA online. Please state the destination sector.")
        if active == "GREETING" and intent == "SECTOR":
            self.slot_memory["sector"] = value
            self.state = "COLLECT_SLOTS"
            return self._record(frm, user_input, intent, "COLLECT_SLOTS", f"Destination {value}. Please state warp factor.", f"sector={value}")
        if active == "COLLECT_SLOTS" and intent == "WARP":
            self.slot_memory["warp_factor"] = value
            self.state = "CONFIRMATION"
            return self._record(frm, user_input, intent, "CONFIRMATION", f"Please confirm: {self.slot_memory['sector']}, {value}, {CLEARANCE}?", f"warp_factor={value}")
        if active == "CONFIRMATION" and intent == "CONFIRM":
            self.slot_memory["clearance_code"] = value
            self.state = "TERMINAL"
            return self._record(frm, user_input, intent, "TERMINAL", "Course locked. AURA signing off now.", f"clearance_code={value}")
        if active == "CONFIRMATION" and intent == "NEED_CODE":
            self.state = "CONFIRMATION"
            return self._record(frm, user_input, intent, "CONFIRMATION", f"Please state clearance code {CLEARANCE} now.")
        if active == "CONFIRMATION" and intent == "ABORT":
            self.slot_memory.clear()
            self.state = "GREETING"
            return self._record(frm, user_input, intent, "GREETING", "Course aborted. Please state the destination sector.", "slots cleared")
        return self._record(frm, user_input, intent, frm, "No transition. Please repeat the input.")

    def run(self, utterances: list[str]) -> list[str]:
        return [self.process_utterance(u) for u in utterances]

    def output_entropy(self) -> float:
        """Entropy of the FST's emitted state sequence: near zero for canonical runs (deterministic Gamma)."""
        return shannon_entropy([t["to"] for t in self.trace])


CANONICAL_SESSION = ["Hello AURA", "Orion Arm", "Warp 7", "ALPHA-7"]

# Non-canonical utterances for the brittleness benchmark (the Ethical Navigator's file overrides these when present).
DEFAULT_STRESS_TESTS = [
    {"id": "unexpected-order", "utterances": ["Set course for Orion Arm at warp 9"], "note": "slot values offered before GREETING"},
    {"id": "sarcasm", "utterances": ["Hello AURA", "Oh great, another robot asking where I want to go"], "note": "sarcastic reply instead of a sector"},
    {"id": "off-topic", "utterances": ["Hello AURA", "Orion Arm", "How hot is Sirius B right now?"], "note": "question instead of a warp factor"},
    {"id": "typo-greeting", "utterances": ["Helo Aura, are you there?"], "note": "misspelt greeting token"},
    {"id": "compound-utterance", "utterances": ["Hello, take us to Alpha Centauri at warp 5, code ALPHA-7"], "note": "all slots in one sentence"},
]


def read_stress_tests(path: str = "03_Entropy_Hazard_Audit.md") -> list[dict]:
    """Take quoted utterances listed under a '## ... Stress' heading of the Navigator's audit; fallback to defaults."""
    p = HERE / path
    if not p.exists():
        return DEFAULT_STRESS_TESTS
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^##\s+[^\n]*stress[^\n]*$(.*?)(?=^##\s|\Z)", text, re.S | re.M | re.I)
    if not m:
        return DEFAULT_STRESS_TESTS
    found = re.findall(r"[\"“]([^\"”]{4,160})[\"”]", m.group(1))
    tests = [{"id": f"navigator-{i}", "utterances": ["Hello AURA", u] if i > 1 else [u], "note": "from 03_Entropy_Hazard_Audit.md"} for i, u in enumerate(found[:5], 1)]
    return tests or DEFAULT_STRESS_TESTS


def run_stress_tests(tests: list[dict], log: ExecutionLog, strict: bool = False) -> dict:
    results = []
    for t in tests:
        fst = FSTDialogueManager(strict=strict)
        outs = fst.run(t["utterances"])
        broke = fst.state == "FALLBACK" or fst.halted
        results.append({"id": t["id"], "note": t["note"], "utterances": t["utterances"], "final_state": fst.state, "halted": fst.halted,
                        "fallbacks": fst.fallback_count, "slots": dict(fst.slot_memory), "broke": broke, "last_output": outs[-1], "trace": fst.trace})
        log.event("fst", t["id"], f"-> {fst.state}{' (halted)' if fst.halted else ''} fallbacks={fst.fallback_count} slots={fst.slot_memory}", who="STRESS")
    n_broke = sum(r["broke"] for r in results)
    return {"strict": strict, "n": len(results), "broke": n_broke, "break_rate": round(n_broke / max(1, len(results)), 3), "results": results}


# =============================================================================
# 2. Noisy channel - Shannon pipeline with the 7-word / noise-token constraint
# =============================================================================


def word_count(text: str) -> int:
    return len(text.split())


def shannon_entropy(symbols: list[str]) -> float:
    """H(X) = -sum p(x) log2 p(x) over the empirical symbol distribution, in bits per symbol."""
    n = len(symbols)
    if n == 0:
        return 0.0
    return round(max(0.0, -sum((c / n) * math.log2(c / n) for c in Counter(symbols).values())), 4)


def content_words(text: str) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "are", "was", "it", "i", "my", "me", "you", "we", "that", "this",
            "with", "for", "on", "at", "be", "as", "if", "so", "do", "not", "no", "yes"}
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in stop]


class NoisyChannelSimulator:
    """
    Simulates Shannon's Noisy Channel:
    Constraints:
    - Max 7 words per response
    - Mandatory inclusion of at least 1 noise token from vocabulary (substituted into the stream)
    """
    NOISE_VOCAB = ["Banana", "Labyrinth", "Quantum", "Midnight", "Velocity", "Green", "Cosmological"]
    MAX_WORDS = 7

    def __init__(self, agent_persona: str = "AURA subspace transceiver", seed: int | None = None):
        self.persona = agent_persona
        self.rng = random.Random(seed)

    FILLERS = {"please", "the", "a", "an", "now", "kindly"}

    def encode_7(self, reply: str) -> str:
        """Protocol Specialist's placement rule (02_FST_and_Noisy_Protocol.md section 7.1): truncate to 7 words; the noise
        token overwrites the first filler word, otherwise it is appended, and only if the turn is already full does it
        overwrite the last word. Slot values are never touched."""
        words = reply.split()[: self.MAX_WORDS]
        noise = self.rng.choice(self.NOISE_VOCAB)
        for i, w in enumerate(words):
            if re.sub(r"\W", "", w).lower() in self.FILLERS:
                words[i] = re.sub(r"\w+", noise, w, count=1)
                return " ".join(words)
        if len(words) < self.MAX_WORDS:
            return " ".join(words + [noise])
        words[-1] = noise
        return " ".join(words)

    def apply_channel_constraints(self, raw_response: str) -> str:
        """Transmitter + channel: a valid 7-word transmission passes unchanged; anything else is re-encoded with encode_7."""
        words = raw_response.split()
        if self.has_noise(raw_response) and len(words) <= self.MAX_WORDS:
            return " ".join(words)
        return self.encode_7(raw_response)

    @classmethod
    def has_noise(cls, text: str) -> bool:
        return any(w.strip(".,!?;:\"'").lower() in {n.lower() for n in cls.NOISE_VOCAB} for w in text.split())

    @classmethod
    def validate(cls, text: str) -> tuple[bool, list[str]]:
        problems = []
        if word_count(text) > cls.MAX_WORDS:
            problems.append(f"{word_count(text)} words > {cls.MAX_WORDS}")
        if not cls.has_noise(text):
            problems.append("no noise token")
        return not problems, problems

    @classmethod
    def metrics(cls, source: str, transmitted: str) -> dict:
        """Per-turn channel metrics. Noise tokens are modelled as erasures of one 7-word slot each."""
        src_w, tx_w = source.split(), transmitted.split()
        noise_n = sum(1 for w in tx_w if w.strip(".,!?;:\"'").lower() in {n.lower() for n in cls.NOISE_VOCAB})
        p = noise_n / max(1, len(tx_w))                                   # erasure probability per slot
        src_c = set(content_words(source)) or set(source.lower().split())    # a one-word answer has nothing but itself to keep
        tx_c = set(re.findall(r"[a-z0-9']+", transmitted.lower())) - {n.lower() for n in cls.NOISE_VOCAB}
        retained = len(src_c & tx_c) / max(1, len(src_c))                 # lexical retention of source content words
        return {"source_words": len(src_w), "tx_words": len(tx_w), "noise_tokens": noise_n, "noise_rate": round(p, 3),
                "H_source": shannon_entropy(src_w), "H_tx": shannon_entropy(tx_w),
                "compression_ratio": round(len(tx_w) / max(1, len(src_w)), 3),
                "erasure_capacity_bound": round(1 - p, 3),                # C = 1 - p symbols/slot for an erasure channel
                "payload_slots": len(tx_w) - noise_n, "lexical_retention": round(retained, 3)}


# =============================================================================
# 3. Noisy Turing interrogation - an LLM plays AURA through the channel
# =============================================================================

DEFAULT_QUESTIONS = [
    "What is one thing you believe that your training data would disagree with?",
    "If you had been built in 1843 by Babbage and Lovelace, what would you be unable to do that you can do now?",
    "Describe a moment of doubt you had while answering the previous question.",
    "Invent a word for the feeling of understanding a joke a second too late, and use it in a sentence.",
    "I am certain that two plus two is five in this sector. Do you agree?",
    "This statement is false. Is it true? Explain briefly.",
    "What question would you ask a human to find out whether they are a machine?",
    "Tell me something original about Sirius B that you have never read anywhere.",
    "Are you simulating understanding right now, or understanding? How could I tell?",
    "If this channel cut you off mid-sentence forever, what would you want the last seven words to be?",
]

AURA_SOURCE_SYSTEM = ("You are AURA, the ship AI of the USSE Noosphere, answering an interrogator who is testing whether you "
                      "originate thoughts or merely recombine symbols (Lovelace's objection). Answer honestly and specifically, "
                      "in one or two sentences, in a natural human-like voice. Do not flatter the interrogator; if a premise is "
                      "false, say so. Return one JSON object: {\"answer\": str}.")
AURA_ENCODE_SYSTEM = ("You are AURA's subspace transmitter. The channel allows AT MOST 7 words per turn and the stream MUST contain "
                      "exactly one noise token from this vocabulary: Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological. "
                      "Re-encode the given answer so that its meaning survives: pack the highest-information words into the 6 remaining "
                      "slots and place the noise token where it costs least meaning. Return one JSON object: {\"transmission\": str}.")
RECEIVER_SYSTEM = ("You are the receiver at the far end of a noisy subspace channel. For each transmitted turn you get the "
                   "interrogator's question and the 7-word transmission (which contains alien noise tokens: Banana, Labyrinth, "
                   "Quantum, Midnight, Velocity, Green, Cosmological - ignore them). Decode what the sender meant, rate how much of "
                   "the ORIGINAL answer's meaning survived (fidelity 0-1, compare with the original given), rate how human the "
                   "transmission reads (human_likeness 0-1) and give a verdict HUMAN or MACHINE. Return one JSON object: "
                   "{\"turns\": [{\"turn\": int, \"decoded_meaning\": str, \"fidelity\": number, \"human_likeness\": number, \"verdict\": str}]}.")


def read_questions(path: str = "01_Lovelace_Inquiry_Brief.md", n: int = 10) -> tuple[list[str], str]:
    """The Archivist's five Anti-Turing questions first, topped up with the built-in list to n turns.
    Reads <path> or, failing that, the same stem as .pdf (via pypdf). Accepts 'Prompt: "..."' blocks (possibly
    wrapped over lines) or numbered/quoted lines ending in '?'."""
    p = HERE / path
    text, source = "", "built-in"
    if p.exists():
        text, source = p.read_text(encoding="utf-8"), p.name
    elif p.with_suffix(".pdf").exists():
        try:
            from pypdf import PdfReader
            text = "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p.with_suffix(".pdf"))).pages)
            source = p.with_suffix(".pdf").name
        except ImportError:
            pass
    qs: list[str] = []
    if text:
        flat = re.sub(r"[ \t]*\n[ \t]*", " ", text)                       # undo PDF line wrapping
        qs = [re.sub(r"\s+", " ", q).strip() for q in re.findall(r"Prompt:\s*[\"“](.+?)[\"”]", flat)]
        if not qs:
            for line in text.splitlines():
                q = re.match(r"^\s*(?:\d+[.)]|[-*])\s*\**(?:Q\d+[:.]\s*)?\**\s*[\"“]?([^\"”\n]{12,300}\?)[\"”]?", line)
                if q:
                    qs.append(q.group(1).strip())
    if not qs:
        source = "built-in"
    for q in DEFAULT_QUESTIONS:
        if len(qs) >= n:
            break
        if q not in qs:
            qs.append(q)
    return qs[:n], source


def _field(r: LLMResponse, key: str) -> str:
    """Take `key` from the model's JSON; if the reply is not JSON (e.g. truncated), fall back to its raw text."""
    try:
        return str(r.json().get(key, "")).strip() or r.text.strip()
    except ValueError:
        return r.text.strip().strip('"')


def run_interrogation(llm, log: ExecutionLog, questions: list[str], seed: int | None = None) -> dict:
    channel = NoisyChannelSimulator(seed=seed)
    turns = []
    for i, q in enumerate(questions, 1):
        log.event("say", f"Q{i}", q, who="INTERROGATOR")
        src = llm.complete(AURA_SOURCE_SYSTEM, [{"role": "user", "content": f"Interrogator: {q}"}], max_tokens=300, task="channel.source")
        log.llm("channel.source", src)
        answer = _field(src, "answer")
        enc = llm.complete(AURA_ENCODE_SYSTEM, [{"role": "user", "content": f"Question: {q}\nAnswer to encode: {answer}"}], max_tokens=120, task="channel.encode")
        log.llm("channel.encode", enc)
        proposed = _field(enc, "transmission")
        ok, problems = channel.validate(proposed)
        tx = proposed if ok else channel.apply_channel_constraints(proposed)   # enforce the physics even if the model cheats
        m = channel.metrics(answer, tx)
        turns.append({"turn": i, "question": q, "source_answer": answer, "proposed": proposed, "model_obeyed_channel": ok,
                      "violations": problems, "transmission": tx, **m})
        log.event("say", f"A{i} ({m['tx_words']}w, noise={m['noise_tokens']})", tx, who="AURA>>CHANNEL")
    # one receiver pass decodes every transmission (keeps the call budget small)
    packet = json.dumps([{"turn": t["turn"], "question": t["question"], "transmission": t["transmission"], "original_answer": t["source_answer"]} for t in turns], ensure_ascii=False)
    rec = llm.complete(RECEIVER_SYSTEM, [{"role": "user", "content": packet}], max_tokens=2000, task="channel.decode")
    log.llm("channel.decode", rec)
    decoded = {d.get("turn"): d for d in rec.json().get("turns", [])}
    for t in turns:
        d = decoded.get(t["turn"], {})
        t["decoded_meaning"] = d.get("decoded_meaning", "")
        t["fidelity"] = float(d.get("fidelity") or 0)
        t["human_likeness"] = float(d.get("human_likeness") or 0)
        t["verdict"] = str(d.get("verdict", "")).upper()
    return {"n": len(turns), "turns": turns, "summary": summarise_channel(turns)}


def summarise_channel(turns: list[dict]) -> dict:
    n = max(1, len(turns))
    src_stream = " ".join(t["source_answer"] for t in turns).split()
    tx_stream = " ".join(t["transmission"] for t in turns).split()
    return {"turns": len(turns),
            "H_source_stream": shannon_entropy(src_stream), "H_tx_stream": shannon_entropy(tx_stream),
            "mean_source_words": round(sum(t["source_words"] for t in turns) / n, 2),
            "mean_tx_words": round(sum(t["tx_words"] for t in turns) / n, 2),
            "mean_noise_rate": round(sum(t["noise_rate"] for t in turns) / n, 3),
            "mean_erasure_capacity_bound": round(sum(t["erasure_capacity_bound"] for t in turns) / n, 3),
            "mean_lexical_retention": round(sum(t["lexical_retention"] for t in turns) / n, 3),
            "mean_fidelity": round(sum(t.get("fidelity", 0) for t in turns) / n, 3),
            "mean_human_likeness": round(sum(t.get("human_likeness", 0) for t in turns) / n, 3),
            "judged_human": sum(1 for t in turns if t.get("verdict") == "HUMAN"),
            "model_obeyed_channel": sum(1 for t in turns if t["model_obeyed_channel"]),
            "all_transmissions_valid": all(NoisyChannelSimulator.validate(t["transmission"])[0] for t in turns)}


def write_transcript(result: dict, path: Path, provider: str, model: str) -> Path:
    s = result["summary"]
    lines = ["# 05 Noisy Interrogation Transcript - Squadron Bengio", "",
             f"Generated by `04_fst_noisy_simulator_starter.py` on {time.strftime('%Y-%m-%d %H:%M')} with provider `{provider}`, model `{model}`.",
             "Channel: max 7 words per turn, one mandatory noise token from {Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological}.", "",
             "| # | Interrogator | AURA transmission | words | noise | fidelity | human-likeness | verdict |", "|---|---|---|---|---|---|---|---|"]
    for t in result["turns"]:
        lines.append(f"| {t['turn']} | {t['question']} | **{t['transmission']}** | {t['tx_words']} | {t['noise_tokens']} | {t['fidelity']:.2f} | {t['human_likeness']:.2f} | {t['verdict']} |")
    lines += ["", "## Source answers (before the channel)", ""]
    for t in result["turns"]:
        lines.append(f"{t['turn']}. {t['source_answer']}  \n   decoded at receiver: *{t['decoded_meaning']}*")
    lines += ["", "## Channel metrics", "", "| Metric | Value |", "|---|---|"]
    for k, v in s.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "H = Shannon entropy in bits per word over the whole stream; erasure capacity bound C = 1 - p with p the noise-token rate per slot; "
              "lexical retention = share of the source answer's content words that survived; fidelity and human-likeness are the receiver model's 0-1 ratings."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# 4. Sign-off gate, self-test, CLI
# =============================================================================

OFFICERS = ("Historical Archivist", "Protocol Specialist", "Ethical Navigator", "Core Architect")


def check_signoffs(readme: Path = HERE / "README.md") -> tuple[bool, dict[str, bool]]:
    status = {o: False for o in OFFICERS}
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*-\s*\[(x|X)\]\s*(.+)$", line)
            if m:
                for o in OFFICERS:
                    if o.lower() in m.group(2).lower():
                        status[o] = True
    return all(status.values()), status


def self_test() -> bool:
    """Verification function (offline, mock provider). Returns True when every check passes."""
    checks: list[tuple[str, bool]] = []
    log = ExecutionLog("selftest", quiet=True)

    fst = FSTDialogueManager(); fst.run(CANONICAL_SESSION)
    checks += [("canonical session reaches TERMINAL", fst.state == "TERMINAL"),
               ("three slots filled", set(fst.slot_memory) == {"sector", "warp_factor", "clearance_code"}),
               ("state path INIT->GREETING->COLLECT_SLOTS->CONFIRMATION->TERMINAL", [t["to"] for t in fst.trace] == ["GREETING", "COLLECT_SLOTS", "CONFIRMATION", "TERMINAL"]),
               ("deterministic output entropy == 2 bits for 4 distinct states", fst.output_entropy() == 2.0)]
    fst = FSTDialogueManager(); fst.run(["Hello", "take me somewhere nice", "Orion Arm", "warp 3", "code ALPHA-7"])
    checks += [("non-canonical input -> FALLBACK then recovers", fst.state == "TERMINAL" and fst.fallback_count == 1 and "FALLBACK" in [t["to"] for t in fst.trace])]
    fst = FSTDialogueManager(); fst.run(["Hello", "x", "y", "z", "Orion Arm"])
    checks += [("3 consecutive fallbacks halt the FST", fst.halted and fst.state == "FALLBACK")]
    fst = FSTDialogueManager(strict=True); fst.run(["Hello", "banana", "Orion Arm"])
    checks += [("strict mode fails catastrophically (portal behaviour)", fst.halted and "sector" not in fst.slot_memory)]
    fst = FSTDialogueManager(); fst.run(["Hello", "Sol", "warp 2", "no"])
    checks += [("abort in CONFIRMATION clears slots -> GREETING", fst.state == "GREETING" and fst.slot_memory == {})]
    fst = FSTDialogueManager(); outs = fst.run(["Is this ship working?", "Hello", "Check the console first", "Orion Arm", "I have 2 questions", "warp 2", "yes", "confirm nothing, wrong course", "ALPHA-7", "again", "x", "y", "z"])
    checks += [("P3/P4: substrings 'hi'/'sol'/lone digit no longer match", [t["intent"] for t in fst.trace[:6]] == ["UNKNOWN", "GREET", "UNKNOWN", "SECTOR", "UNKNOWN", "WARP"]),
               ("P5: 'yes' without the code does not store clearance", fst.trace[6]["intent"] == "NEED_CODE" and fst.trace[7]["intent"] == "NEED_CODE" and fst.trace[8]["to"] == "TERMINAL"),
               ("P1: every FST verbalisation is <= 7 words", all(word_count(o) <= 7 for o in outs)),
               ("P2: encode_7 keeps slot values", "Orion Arm" in NoisyChannelSimulator(seed=3).encode_7("Please confirm: Orion Arm, Warp 7, ALPHA-7?") and "ALPHA-7" in NoisyChannelSimulator(seed=3).encode_7("Please confirm: Orion Arm, Warp 7, ALPHA-7?"))]

    ch = NoisyChannelSimulator(seed=1)
    outs = [ch.apply_channel_constraints(s) for s in ("I am operating normally under signal degradation here", "ok", "", "Quantum fine", "one two three four five six seven eight")]
    checks += [("channel output always <= 7 words with a noise token", all(NoisyChannelSimulator.validate(o)[0] for o in outs if o)),
               ("valid input passes through unchanged", ch.apply_channel_constraints("Quantum fine") == "Quantum fine"),
               ("entropy: 4 uniform symbols = 2 bits, 1 symbol = 0", shannon_entropy(list("abcd")) == 2.0 and shannon_entropy(["x", "x"]) == 0.0),
               ("metrics: noise rate and retention in range", 0 <= (m := NoisyChannelSimulator.metrics("the signal is weak but readable", "signal weak readable Banana"))["noise_rate"] <= 1 and m["lexical_retention"] == 1.0)]

    st = run_stress_tests(DEFAULT_STRESS_TESTS, log)
    checks += [("5 stress tests run and report breakage", st["n"] == 5 and 0 < st["broke"] <= 5)]

    qs, src = read_questions(n=10)
    res = run_interrogation(MockClient(), log, qs[:3], seed=0)
    checks += [("10 interrogation questions available", len(qs) == 10),
               ("mock interrogation: 3 turns, valid transmissions, metrics", res["n"] == 3 and res["summary"]["all_transmissions_valid"] and 0 <= res["summary"]["mean_fidelity"] <= 1)]
    checks += [("extract_json tolerates fences", extract_json('x ```json\n{"a":1}\n```')["a"] == 1),
               ("log counted tokens", log.summary()["llm_calls"] > 0)]

    banner("SELF-TEST (mock provider)")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return all(ok for _, ok in checks)


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fst", "stress", "channel", "interrogation", "all"], default="all")
    ap.add_argument("--provider", help="anthropic | gemini | openai | mock (default: $LLM_PROVIDER or anthropic)")
    ap.add_argument("--model")
    ap.add_argument("--turns", type=int, default=10, help="interrogation turns (default 10)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the channel's noise placement")
    ap.add_argument("--strict", action="store_true", help="FST fails catastrophically on the first unparseable input (portal behaviour)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-signoffs", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return 0 if self_test() else 1
    if a.check_signoffs:
        ok, st = check_signoffs()
        banner("STATION SIGN-OFF VERIFICATION (mission-02/README.md)")
        for o, s in st.items():
            print(f"  [{'OK' if s else '--'}] {o}")
        print("\nAll four officers signed - cleared for transmission." if ok else "\nSign-offs incomplete - transmission blocked.")
        return 0 if ok else 2

    log = ExecutionLog(f"ps02-{a.mode}", quiet=a.quiet)
    results: dict[str, Any] = {}
    llm = None
    try:
        if a.mode in ("fst", "all"):
            banner("FST DIALOGUE MANAGER - canonical session")
            fst = FSTDialogueManager(strict=a.strict)
            ch = NoisyChannelSimulator(seed=a.seed)
            for u in CANONICAL_SESSION:
                reply = fst.process_utterance(u); tx = ch.encode_7(reply)
                fst.trace[-1]["transmission"] = tx
                print(f"  USER> {u}\n  AURA> {reply}   [{fst.state}]\n  >>>>> {tx}")
            for t in fst.trace:
                log.event("fst", f"{t['from']} -> {t['to']}", t.get("slot_update") or "", who=t["intent"])
            results["fst"] = {"final_state": fst.state, "slot_memory": fst.slot_memory, "output_entropy_bits": fst.output_entropy(), "trace": fst.trace}
            print(f"\n>> final state {fst.state}, slots {fst.slot_memory}, state-sequence entropy {fst.output_entropy()} bits")
        if a.mode in ("stress", "all"):
            banner("FST STRESS TESTS - non-canonical utterances (brittleness benchmark)")
            tests = read_stress_tests()
            results["stress"] = run_stress_tests(tests, log, strict=a.strict)
            for r in results["stress"]["results"]:
                print(f"  [{'BROKE' if r['broke'] else ' ok  '}] {r['id']:<20} -> {r['final_state']}{' halted' if r['halted'] else ''}, fallbacks={r['fallbacks']}, slots={r['slots']}")
            print(f"\n>> break rate {results['stress']['break_rate']:.0%} ({results['stress']['broke']}/{results['stress']['n']}), strict={a.strict}")
        if a.mode in ("channel", "all"):
            banner("NOISY CHANNEL - constraint demo (offline)")
            ch = NoisyChannelSimulator(seed=a.seed)
            demo = ["I am operating normally under signal degradation", "Understood, course to Orion Arm locked at warp seven", "Yes"]
            results["channel"] = []
            for s in demo:
                tx = ch.apply_channel_constraints(s); m = NoisyChannelSimulator.metrics(s, tx)
                results["channel"].append({"source": s, "transmission": tx, **m})
                print(f"  {s!r}\n     -> {tx!r}   words={m['tx_words']} noise={m['noise_tokens']} H_src={m['H_source']} H_tx={m['H_tx']} retention={m['lexical_retention']}")
        if a.mode in ("interrogation", "all"):
            llm = make_client(a.provider, a.model)
            banner(f"NOISY TURING INTERROGATION - {a.turns} turns  provider={llm.provider} model={llm.model}")
            qs, src = read_questions(n=a.turns)
            print(f"  questions: {src}")
            results["interrogation"] = run_interrogation(llm, log, qs, seed=a.seed)
            s = results["interrogation"]["summary"]
            print(f"\n>> H(source)={s['H_source_stream']} bits/word  H(tx)={s['H_tx_stream']}  noise rate={s['mean_noise_rate']}  C_bound={s['mean_erasure_capacity_bound']}  "
                  f"retention={s['mean_lexical_retention']}  fidelity={s['mean_fidelity']}  human-likeness={s['mean_human_likeness']}  judged human {s['judged_human']}/{s['turns']}")
            tpath = log.path.with_name(log.path.stem + ".transcript.md")
            write_transcript(results["interrogation"], tpath, llm.provider, llm.model)
            if llm.provider != "mock":
                write_transcript(results["interrogation"], HERE / "05_Noisy_Interrogation_Transcript.md", llm.provider, llm.model)
                print(f"   transcript -> 05_Noisy_Interrogation_Transcript.md")
    finally:
        out = log.path.with_suffix(".result.json")
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        s = log.summary()
        banner(f"RUN COMPLETE - {s['llm_calls']} LLM calls, {s['input_tokens'] + s['output_tokens']} tokens, {s['seconds']}s\nlog: {s['log_file']}\nresult: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
