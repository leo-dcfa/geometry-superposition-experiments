"""Ordered-alternative trend tests — the replacement for falsifier F1.1 (D14).

F1.1 as pre-registered killed P1 on **any** adjacent-pair inversion in N*(κ).
The 00e power analysis measured that criterion firing with probability 0.533
when H-MAIN is exactly true: adjacent true effects are separated by 0.09–0.34
in log units against comparable seed noise, so asking every pair to order
correctly is asking noise to behave. Reaching a 10% false-rejection rate would
have needed 58 seeds per cell, against 3 for a contrast testing the same
claim.

What P1 actually asserts is a *trend*: N* decreases as κ increases. That is
what these test, at a defensible cost.

- ``jonckheere_terpstra`` — the standard distribution-free test against an
  ordered alternative across k groups. Primary.
- ``spearman_midrank`` — Spearman's ρ with midrank ties, the tie policy SPEC
  §8 requires after the parent program's index-order tie-breaking made a
  headline correlation uninterpretable. Reported alongside.
- ``extreme_pair_contrast`` — the two-arm comparison between the grid's
  endpoints, which 00e showed needs only 3 seeds.

All p-values come from permutation nulls rather than asymptotic
approximations: the group sizes here are small (5–10 seeds), which is exactly
where the normal approximation to JT is least trustworthy.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PERMUTATIONS = 20_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def jt_statistic(groups: list[list[float]]) -> float:
    """Jonckheere–Terpstra statistic for a *decreasing* alternative.

    Counts, over every ordered pair of groups (i < j), how often a value in
    the later group falls below one in the earlier group. Ties score ½. Larger
    means a stronger decreasing trend.
    """
    total = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i], dtype=float)[:, None]
            b = np.asarray(groups[j], dtype=float)[None, :]
            total += float((b < a).sum() + 0.5 * (b == a).sum())
    return total


def jonckheere_terpstra(
    groups: list[list[float]],
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict:
    """Permutation test against the ordered (decreasing) alternative.

    ``groups`` must already be ordered by the independent variable — for P1,
    by increasing κ, so that H-MAIN predicts a decreasing response.
    """
    sizes = [len(g) for g in groups]
    if len(groups) < 3:
        raise ValueError("a trend test needs at least three ordered groups")
    observed = jt_statistic(groups)

    pool = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    rng = _rng(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pool)
        split, out = 0, []
        for n in sizes:
            out.append(pool[split : split + n])
            split += n
        if jt_statistic(out) >= observed:
            count += 1
    # add-one correction: a permutation p-value is never exactly 0
    p_value = (count + 1) / (n_permutations + 1)

    n_total = sum(sizes)
    max_stat = (n_total**2 - sum(s**2 for s in sizes)) / 2
    return {
        "statistic": observed,
        "max_statistic": max_stat,
        "normalized": observed / max_stat if max_stat else float("nan"),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "alternative": "decreasing across groups in the given order",
        "group_sizes": sizes,
    }


def spearman_midrank(x: list[float], y: list[float]) -> dict:
    """Spearman's ρ with midrank ties, plus the tied-mass diagnostic SPEC §8 wants.

    The parent program's −0.822 full-vocab correlation was made
    uninterpretable by index-order tie-breaking; midranks are the fix, and the
    tied fraction is reported so a reader can see when it matters.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xs.size != ys.size:
        raise ValueError("x and y must be the same length")

    def midrank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=float)
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return ranks

    rx, ry = midrank(xs), midrank(ys)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    tied_x = float(1 - np.unique(xs).size / xs.size)
    tied_y = float(1 - np.unique(ys).size / ys.size)
    return {
        "rho": rho,
        "tie_policy": "midranks",
        "tied_mass_x": tied_x,
        "tied_mass_y": tied_y,
        "tied_mass_exceeds_10pct": bool(max(tied_x, tied_y) > 0.10),
    }


def extreme_pair_contrast(
    low_group: list[float], high_group: list[float], n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict:
    """Two-sample permutation contrast between the endpoints of the grid.

    00e: this needs 3 seeds where F1.1's all-pairs criterion needed 58, for
    the same underlying claim.
    """
    a, b = np.asarray(low_group, dtype=float), np.asarray(high_group, dtype=float)
    observed = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = _rng(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pool)
        if float(pool[: a.size].mean() - pool[a.size :].mean()) >= observed:
            count += 1
    return {
        "difference": observed,
        "p_value": (count + 1) / (n_permutations + 1),
        "alternative": "low-kappa group exceeds high-kappa group",
        "n_permutations": n_permutations,
    }
