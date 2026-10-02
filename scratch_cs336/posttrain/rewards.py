"""Answer extraction and grading for GSM8K-style numeric tasks (the subset of hw5's ``drgrpo_grader`` GSM8K needs).

Answers are compared as exact rationals after normalization (units, ``$``, ``%``, thousands separators, ``\\frac``);
non-numeric answers fall back to case- and whitespace-insensitive string equality.
"""

from __future__ import annotations

import re
from fractions import Fraction

from scratch_cs336.posttrain.losses import RewardFn

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_THOUSANDS = re.compile(r"(\d),(\d{3})(?=$|\D)")
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_UNITS = re.compile(
    r"(?:dollars?|cents?|percent|degrees?|cm|centimeters?|meters?|miles?|seconds?|minutes?|hours?|days?|weeks?"
    r"|months?|years?|foot|feet|inch(?:es)?|yards?|units?)(?:\^\d+)?\b"
)


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces, or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    return None


def gsm8k_ground_truth(answer: str) -> str:
    """Final answer of a GSM8K ``answer`` field: the text after ``####``, without thousands separators."""
    _, sep, final = answer.rpartition("####")
    if not sep:
        raise ValueError(f"GSM8K answer has no '####' marker: {answer[-80:]!r}")
    return final.strip().replace(",", "")


def parse_last_number(text: str) -> str | None:
    """Last number in free text (hw5 ``parse_gsm8k_response``), thousands separators removed."""
    numbers = _NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def normalize_answer(answer: str) -> str:
    """Canonical string form: a reduced rational (``"18"``, ``"-1/2"``) when numeric, else lowercased text."""
    s = answer.strip()
    s = _TEXT.sub(r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\$", "").replace("$", "").replace("\\%", "")
    s = s.replace("%", "").replace("\\left", "").replace("\\right", "")
    s = _FRAC.sub(r"(\1)/(\2)", s)
    s = _UNITS.sub("", s)
    while True:
        unsplit = _THOUSANDS.sub(r"\1\2", s)
        if unsplit == s:
            break
        s = unsplit
    s = s.replace(" ", "").replace("{", "").replace("}", "").rstrip(".")
    if s.startswith("(") and s.endswith(")") and s.count("(") == 1:
        s = s[1:-1]
    value = _to_fraction(s)
    return s.lower() if value is None else str(value)


def _to_fraction(s: str) -> Fraction | None:
    num, sep, den = s.partition("/")
    try:
        if not sep:
            return Fraction(s)
        value = Fraction(num.strip("()")) / Fraction(den.strip("()"))
    except (ValueError, ZeroDivisionError):
        return None
    return value


def grade(model_answer: str, ground_truth: str) -> bool:
    if "\\boxed" in ground_truth:
        boxed = last_boxed(ground_truth)
        ground_truth = boxed if boxed is not None else ground_truth
    return normalize_answer(model_answer) == normalize_answer(ground_truth)


def _scored(formatted: bool, correct: bool) -> dict[str, float]:
    # A formatted but wrong answer earns nothing: no format reward, to avoid reward hacking (as in hw5).
    return {"format_reward": float(formatted), "answer_reward": float(correct), "reward": float(correct)}


def r1_zero_reward(response: str, ground_truth: str) -> dict[str, float]:
    """The r1_zero prompt ends in ``<think>``; a formatted response contains ``</think> <answer>`` and ``</answer>``."""
    if "</think> <answer>" not in response or "</answer>" not in response:
        return _scored(False, False)
    answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in answer:
        boxed = last_boxed(answer)
        if boxed is None:
            return _scored(True, False)
        answer = boxed
    return _scored(True, grade(answer, ground_truth))


def boxed_reward(response: str, ground_truth: str) -> dict[str, float]:
    """For prompts asking for ``\\boxed{}`` (hw5 ``question_only``): unparseable → 0 format reward."""
    answer = last_boxed(response)
    if answer is None:
        return _scored(False, False)
    return _scored(True, grade(answer, ground_truth))


def last_number_reward(response: str, ground_truth: str) -> dict[str, float]:
    """Lenient grading for chat prompts: the last number in the response is the answer."""
    answer = parse_last_number(response)
    if answer is None:
        return _scored(False, False)
    return _scored(True, grade(answer, ground_truth))


REWARD_FNS: dict[str, RewardFn] = {
    "r1_zero": r1_zero_reward,
    "boxed": boxed_reward,
    "last_number": last_number_reward,
}
