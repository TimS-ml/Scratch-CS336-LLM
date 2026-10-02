import pytest

from scratch_cs336.posttrain.rewards import (
    boxed_reward,
    grade,
    gsm8k_ground_truth,
    last_boxed,
    last_number_reward,
    parse_last_number,
    r1_zero_reward,
)

RIGHT = {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
WRONG = {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
UNFORMATTED = {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}


def test_gsm8k_ground_truth_takes_final_answer_without_separators():
    assert gsm8k_ground_truth("She makes 9 * 2 = $<<9*2=18>>18.\n#### 1,018") == "1018"
    with pytest.raises(ValueError):
        gsm8k_ground_truth("no marker")


def test_last_number_parse():
    text = "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May."
    assert parse_last_number(text) == "72"
    assert parse_last_number("a total of 1,234,567 dollars, or -3.5 each") == "-3.5"
    assert parse_last_number("seventy-two clips") is None


def test_last_boxed_balances_nested_braces():
    assert last_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{unclosed") is None


@pytest.mark.parametrize(
    ("given", "truth"),
    [("18", "18"), ("$18.00", "18"), ("1,018", "1018"), (r"\frac{1}{2}", "0.5"), ("50%", "50"), ("18 dollars", "18"),
     (r"\text{18}", "18"), ("-4", "-4"), ("3/6", "1/2")],
)  # fmt: skip
def test_grade_accepts_equal_numbers(given, truth):
    assert grade(given, truth)


@pytest.mark.parametrize(("given", "truth"), [("17", "18"), ("1018", "10.18"), ("-4", "4"), ("", "18")])
def test_grade_rejects_different_numbers(given, truth):
    assert not grade(given, truth)


def test_r1_zero_reward_requires_format():
    assert r1_zero_reward("reasoning </think> <answer> 72 </answer>", "72") == RIGHT
    assert r1_zero_reward(r"r </think> <answer> \boxed{72} </answer>", "72") == RIGHT
    assert r1_zero_reward("reasoning </think> <answer> 71 </answer>", "72") == WRONG
    assert r1_zero_reward("reasoning </think><answer> 72 </answer>", "72") == UNFORMATTED
    assert r1_zero_reward("reasoning </think> <answer> 72", "72") == UNFORMATTED


def test_boxed_and_last_number_rewards():
    assert boxed_reward(r"so \boxed{1,000}.", "1000") == RIGHT
    assert boxed_reward("so 1000", "1000") == UNFORMATTED
    assert last_number_reward("The answer is 1,000.", "1000") == RIGHT
    assert last_number_reward("The answer is unknown", "1000") == UNFORMATTED
