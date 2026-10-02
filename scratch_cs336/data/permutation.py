"""Stateless seeded bijection on ``[0, n)``: a Feistel network over the next even power of two + cycle-walking."""

from __future__ import annotations

_MASK64 = (1 << 64) - 1
_ROUNDS = 8


def _mix(x: int) -> int:
    """splitmix64 finalizer."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def derive_seed(seed: int, *salts: int) -> int:
    """Independent 64-bit seed for each ``(seed, *salts)``, e.g. one permutation per epoch."""
    state = _mix(seed & _MASK64)
    for salt in salts:
        state = _mix(state ^ _mix(salt & _MASK64))
    return state


class FeistelPermutation:
    """``perm(i)`` for ``i in [0, n)`` is a bijection onto ``[0, n)``, O(1) in ``n`` and without any table."""

    def __init__(self, n: int, seed: int):
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        self.n = n
        half_bits = max(((n - 1).bit_length() + 1) // 2, 1)  # 2 * half_bits >= bits needed for n - 1
        self._half_bits = half_bits
        self._half_mask = (1 << half_bits) - 1
        self._keys = [derive_seed(seed, r) for r in range(_ROUNDS)]

    def _feistel(self, x: int) -> int:
        left, right = x >> self._half_bits, x & self._half_mask
        for key in self._keys:
            left, right = right, left ^ (_mix(right ^ key) & self._half_mask)
        return (left << self._half_bits) | right

    def __call__(self, i: int) -> int:
        if not 0 <= i < self.n:
            raise IndexError(f"{i} outside [0, {self.n})")
        x = self._feistel(i)
        while x >= self.n:  # cycle-walk: the domain is < 4n, so this takes < 4 steps in expectation
            x = self._feistel(x)
        return x
