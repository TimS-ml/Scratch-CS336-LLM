from __future__ import annotations

import pytest

from scratch_cs336.data.permutation import FeistelPermutation


@pytest.mark.parametrize("n", [1, 2, 3, 7, 100, 1000, 4097, 12345])
def test_is_a_bijection_for_any_n(n: int):
    perm = FeistelPermutation(n, seed=3)
    assert sorted(perm(i) for i in range(n)) == list(range(n))


def test_depends_on_seed_and_is_deterministic():
    a, b = FeistelPermutation(1000, seed=1), FeistelPermutation(1000, seed=2)
    assert [a(i) for i in range(1000)] == [FeistelPermutation(1000, seed=1)(i) for i in range(1000)]
    assert [a(i) for i in range(1000)] != [b(i) for i in range(1000)]


def test_actually_shuffles():
    perm = FeistelPermutation(1000, seed=0)
    assert sum(perm(i) == i for i in range(1000)) < 20
    assert abs(sum(perm(i) for i in range(100)) / 100 - 500) < 150  # prefix is spread over the whole range


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        FeistelPermutation(10, seed=0)(10)
