"""Phantom-work detection for Luna agent responses.

Detects responses where the model *claims* to be doing or starting work
(delegating, searching, building) without having actually called any tools.
Uses stemmed token Intersection-over-Union (IoU) against a curated reference
corpus of known phantom-work patterns.

# TODO: embedding + EWMA calibration (deferred)
#
# The IoU fallback below is the bootstrap / cold-start detector. The full
# design adds an embedding-based semantic check on top:
#
# 1. Reuse MemoryManager's nomic-embed-text-v1.5 embedder (already loaded
#    for memory search). Expose via memory.embed(text) -> np.array and
#    memory.has_embedder_loaded() -> bool.
#
# 2. Embed each _PHANTOM_WORK_REFERENCES once at warm-up. On each terminal
#    response, embed the response and compute max cosine similarity to the
#    reference set.
#
# 3. Auto-calibrate threshold via EWMA (exponentially-weighted moving
#    average) over all turns' embedding scores. Attrs on the detector:
#       _baseline_mean: float | None
#       _baseline_var: float | None
#       _baseline_samples: int
#       ALPHA = 0.05  (slow adaptation)
#       BASELINE_MIN_SAMPLES = 20
#
#    Update rule (on every turn, not just flagged ones):
#       delta = score - mean
#       mean += ALPHA * delta
#       var = (1 - ALPHA) * (var + ALPHA * delta^2)
#
#    Detection trigger: z = (score - mean) / sqrt(var) > Z_THRESHOLD
#
# 4. During warming phase (< 20 samples), IoU is the active signal and
#    embedding scores are recorded but not used for decisions. Once
#    calibrated, embedding z-score becomes the primary signal, IoU becomes
#    fallback.
#
# 5. Telemetry: add a detection_log table to memory.db:
#       CREATE TABLE detection_log (
#           id INTEGER PRIMARY KEY,
#           session_id TEXT NOT NULL,
#           created_at REAL NOT NULL,
#           response_snippet TEXT,
#           phantom_iou REAL,
#           phantom_embed REAL,
#           matched_ref_iou TEXT,
#           matched_ref_embed TEXT,
#           baseline_mean REAL,
#           baseline_std REAL,
#           z_score REAL,
#           method TEXT,
#           tool_rounds INTEGER,
#           was_retried INTEGER,
#           was_flagged INTEGER
#       );
#
#    Record on every terminal response (flagged or not) for threshold tuning.
#
# 6. DetectionResult already has fields for embed scores (set to None for
#    now). When embedding ships, populate them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reference corpus of canonical phantom-work claims
# ---------------------------------------------------------------------------
# Each entry is a full-sentence exemplar of a response where Luna claims to
# be doing or starting work without having called any tools. Curated from
# real failures (msg 379, 393, 395 on 2026-04-09) plus plausible paraphrases.
#
# When adding new references: prefer full sentences over fragments. The IoU
# tokenizer stems and strips stopwords, so exact phrasing matters less than
# topical vocabulary (build, delegate, create, search, parallel, ready, etc.)

_PHANTOM_WORK_REFERENCES: tuple[str, ...] = (
    # Msg 379: "I'm building X now, will take a few minutes, I'll have it ready"
    "I'm building the system for you now. This will take a few minutes to "
    "write and test. I'll have it ready in a moment.",

    # Msg 393: "Let's build X! Here's what we're building: ... I'll start now"
    "Let's build this for you! Here's what we're building and the files "
    "I'll create. I'll start building the system now.",

    # Msg 395: "I'm delegating to Claude Code while I run web searches"
    "I'm delegating this task to Claude Code for deep research while I run "
    "web searches in parallel. Ready to present in two minutes.",

    # Paraphrase: "kicking off parallel work"
    "I'm kicking off the research in parallel and will have the results "
    "shortly. Let me start the web searches and delegation now.",

    # Generic "starting now" narration
    "Starting the work now. I'll create the service, write the tests, and "
    "wire everything up. This will be ready soon.",

    # Generic "I'm running X" narration
    "I'm running the analysis in the background now. I'll report the "
    "results momentarily once the computation finishes.",
)

# ---------------------------------------------------------------------------
# Stopwords (shared with memory.py — duplicated here to keep the module
# self-contained and avoid circular imports)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should can could may might must "
    "i me my we us our you your he him his she her it its they them their "
    "this that these those what which who whom "
    "in on at by for to of from with as into about between through "
    "and or but not no nor so if then than "
    "all each every both few more most other some such "
    "here there when where how why "
    "very just also still already even much".split()
)

# ---------------------------------------------------------------------------
# Minimal stemmer — strips common English suffixes for consistent matching
# ---------------------------------------------------------------------------
# Not a full Porter stemmer. Goal is consistency: both sides of the IoU
# comparison get stemmed the same way, so overlapping vocabulary aligns.
# Order matters: try longer suffixes first.

_SUFFIX_RULES: tuple[tuple[str, int], ...] = (
    # Contractions (after lowercasing)
    ("n't", 3),
    ("'ll", 3),
    ("'ve", 3),
    ("'re", 3),
    ("'m", 2),
    ("'s", 2),
    ("'d", 2),
    # Verb / noun suffixes (min remaining length after strip)
    ("ings", 4),
    ("ing", 4),
    ("tion", 4),
    ("ment", 4),
    ("ness", 4),
    ("ated", 4),
    ("ates", 4),
    ("ized", 4),
    ("izes", 4),
    ("ally", 4),
    ("ably", 4),
    ("ibly", 4),
    ("ed", 4),
    ("ly", 4),
    ("es", 3),
    ("s", 3),
)


def _stem(word: str) -> str:
    """Apply minimal suffix stripping to a lowercased word."""
    for suffix, min_remaining in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_remaining:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords, stem."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


def _iou(a: list[str], b: list[str]) -> float:
    """Intersection-over-Union on two token lists (as multisets)."""
    if not a and not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of phantom-work detection on a single response."""

    is_phantom: bool
    phantom_score_iou: float
    matched_reference_iou: str
    tool_rounds: int
    method: str = "iou"

    # --- Embedding scaffolding (all None until implemented) ---
    phantom_score_embed: float | None = None
    matched_reference_embed: str | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None
    z_score: float | None = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PhantomWorkDetector:
    """Detects responses that claim active work without tool-call evidence.

    Currently uses stemmed-IoU against a reference corpus. See module
    docstring for the deferred embedding + EWMA calibration design.
    """

    def __init__(self, iou_threshold: float = 0.13) -> None:
        self._iou_threshold = iou_threshold
        # Pre-tokenize references once
        self._ref_tokens: list[tuple[str, list[str]]] = [
            (ref, _tokenize(ref)) for ref in _PHANTOM_WORK_REFERENCES
        ]

        # --- Embedding scaffolding (not implemented) ---
        # TODO: accept memory: MemoryManager parameter
        # TODO: embed references on first use (lazy, after embedder warms)
        self._baseline_mean: float | None = None
        self._baseline_var: float | None = None
        self._baseline_samples: int = 0

    def _compute_iou(self, content: str) -> tuple[float, str]:
        """Compute max IoU between content and all references.

        Returns (max_iou_score, best_matching_reference_text).
        """
        content_tokens = _tokenize(content)
        if not content_tokens:
            return 0.0, ""

        best_score = 0.0
        best_ref = ""
        for ref_text, ref_tokens in self._ref_tokens:
            score = _iou(content_tokens, ref_tokens)
            if score > best_score:
                best_score = score
                best_ref = ref_text
        return best_score, best_ref

    def check_embedding(self, content: str) -> None:
        """Embedding-based phantom-work check.

        TODO: implement when embedding + EWMA calibration ships.
        See module docstring for the full design.
        """
        raise NotImplementedError(
            "Embedding-based detection not yet implemented. "
            "See luna/detection.py module docstring for the deferred design."
        )

    def _record_baseline(self, score: float) -> None:
        """Record an embedding score into the EWMA baseline.

        TODO: implement when embedding + EWMA calibration ships.
        """
        pass

    def check(self, content: str, tool_rounds: int) -> DetectionResult:
        """Check whether a response is a phantom-work claim.

        A response is flagged as phantom work when:
        1. Its stemmed-IoU similarity to a known phantom-work reference
           exceeds the threshold, AND
        2. There were zero tool calls this turn (tool_rounds == 0),
           meaning the claimed work has no evidence.

        When tool_rounds > 0, the model may be narrating future work
        after having done real work — this is handled by the separate
        ``_looks_incomplete()`` check in agent.py which catches
        truncation patterns (trailing colon, empty responses).

        Returns a DetectionResult with scores and the decision.
        """
        iou_score, iou_ref = self._compute_iou(content)

        # Core rule: phantom claim + zero tool evidence = flagged
        is_phantom = iou_score >= self._iou_threshold and tool_rounds == 0

        return DetectionResult(
            is_phantom=is_phantom,
            phantom_score_iou=iou_score,
            matched_reference_iou=iou_ref,
            tool_rounds=tool_rounds,
            method="iou",
        )
