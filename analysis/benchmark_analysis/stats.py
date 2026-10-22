# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimators, intervals and tests (docs/research/METHODOLOGY.md §4; RESULTS_RESEARCH RD-2 to RD-16, RD-52,
RD-53). Pure functions of plain numbers.

- Proportions: Wilson; the discordant split of paired success with Clopper–Pearson and the exact conditional
  McNemar test; the paired difference in rates with Tango's score interval (descriptive).
- Times restricted to L: Kaplan–Meier with the midpoint quantile rule (RD-10), the order-statistic interval of
  the median, RMST(L) with a t interval.
- Paired means (ΔR, log ratios): the exact sign-flip test, its inversion below 10 blocks and a percentile block
  bootstrap from 10 (RD-5, RD-11); the Monte Carlo test above 20 pairs.
- Paired Hodges–Lehmann: the Wilcoxon signed-rank test with midranks and Pratt's zeros, exact conditional up to
  40 pairs and normal above, its interval by inverting the same test (RD-12).
- Holm over the plan's family (RD-15).

When the requested level cannot be reached with so few pairs, an interval is computed at the highest level the
data allow and says so; a bootstrap below 10 clusters is refused.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

BOOTSTRAP_MIN_CLUSTERS = 10  # RD-5
EXACT_SIGN_FLIP_PAIRS = 20  # exact enumeration of 2^n sign patterns up to here, Monte Carlo above
EXACT_SIGNED_RANK_PAIRS = 40  # RD-12
TOLERANCE = 1e-9

Floats = NDArray[np.float64]
Values = Sequence[float] | Floats


@dataclass(frozen=True, slots=True)
class Interval:
    # None: "> L" (the flag below), unbounded or not defined (the note says which)
    lower: float | None
    upper: float | None
    level: float  # the level reached; below the requested level only when the data allow no more
    method: str
    note: str | None = None
    lower_beyond_limit: bool = False  # the limit is a run censored at L: report it as "> L"
    upper_beyond_limit: bool = False


BOOTSTRAP_STREAM, SIGN_FLIP_STREAM, POOLED_BOOTSTRAP_STREAM, POOLED_SIGN_FLIP_STREAM = 1, 2, 3, 4
JITTER_STREAM = 5  # horizontal jitter of points in figures (display only)


def random_stream(seed: int, purpose: int) -> np.random.Generator:
    """An independent, reproducible stream per purpose from the plan's one seed."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(purpose,)))


def finite(values: Values) -> Floats:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a one-dimensional sequence of finite numbers")
    return array


# Proportions -----------------------------------------------------------------------------------------------


def wilson(successes: int, trials: int, alpha: float) -> Interval:
    """Wilson score interval of k/n."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p, n = successes / trials, trials
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Interval(max(0.0, centre - half), min(1.0, centre + half), 1 - alpha, "Wilson")


def clopper_pearson(successes: int, trials: int, alpha: float) -> Interval:
    """Exact (Clopper–Pearson) interval of k/n."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return Interval(lower, upper, 1 - alpha, "Clopper–Pearson")


def mcnemar_exact(only_profile: int, only_baseline: int) -> float:
    """Exact conditional McNemar p-value: the discordant split against Bin(n10 + n01, 1/2), two-sided (RD-52).
    At π = 1/2 it equals twice the smaller tail, so it agrees with the Clopper–Pearson interval of the split."""
    discordant = only_profile + only_baseline
    if discordant == 0:
        return 1.0
    tail = float(stats.binom.cdf(min(only_profile, only_baseline), discordant, 0.5))
    return min(1.0, 2 * tail)


def tango(only_profile: int, only_baseline: int, pairs: int, alpha: float) -> Interval:
    """Tango's score interval for the paired difference in rates (n10 − n01)/n (Tango 1998; the closed-form
    constrained estimate as given by Fagerland, Lydersen and Laake 2014). Descriptive (RD-53)."""
    b, c, n = only_profile, only_baseline, pairs
    if n <= 0 or b < 0 or c < 0 or b + c > n:
        raise ValueError("need 0 <= n10 + n01 <= n and n > 0")
    z = float(stats.norm.ppf(1 - alpha / 2))
    estimate = (b - c) / n

    def score(delta: float) -> float:
        a_term, b_term = 2 * n, -b - c + (2 * n - b + c) * delta
        c_term = -c * delta * (1 - delta)
        root = max(0.0, b_term * b_term - 4 * a_term * c_term)
        p21 = (math.sqrt(root) - b_term) / (2 * a_term)
        variance = n * (2 * p21 + delta * (1 - delta))
        if variance <= 0:
            return math.copysign(math.inf, b - c - n * delta) if b - c != n * delta else 0.0
        return (b - c - n * delta) / math.sqrt(variance)

    def bound(target: float, edge: float) -> float:
        # score() falls as delta rises and is 0 at the estimate: the limit is where it crosses ±z between the
        # estimate and the edge of [-1, 1], or that edge when it never does
        inner = edge - math.copysign(1e-12, edge)
        if (score(inner) - target) * (score(estimate) - target) > 0:
            return edge
        low, high = sorted((estimate, inner))
        return float(optimize.brentq(lambda d: score(d) - target, low, high, xtol=1e-13))

    lower = bound(z, -1.0) if estimate > -1 else -1.0
    upper = bound(-z, 1.0) if estimate < 1 else 1.0
    return Interval(lower, upper, 1 - alpha, "Tango score")


def holm(pvalues: Values) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in input order."""
    p = finite(pvalues)
    adjusted = np.empty(p.size)
    running = 0.0
    for position, index in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (p.size - position) * float(p[index])))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def exact_floor(family_size: int, alpha: float) -> int:
    """The fewest nonzero differences (or discordant blocks) with which an exact two-sided test can reject
    within a Holm family of m: 1 + log2(m/α), rounded up (RESULTS_RESEARCH §5.3)."""
    return math.ceil(1 + math.log2(max(1, family_size) / alpha) - TOLERANCE)


# Restricted times ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Survival:
    """Kaplan–Meier step function: Ŝ is ``survival[i]`` from ``times[i]`` until the next time."""

    times: tuple[float, ...]  # distinct event times
    survival: tuple[float, ...]
    at_risk: tuple[int, ...]
    events: tuple[int, ...]
    n: int
    last_time: float  # the largest observed time, event or censored: the curve ends there


def kaplan_meier(times: Values, events: Sequence[bool] | NDArray[np.bool_]) -> Survival:
    """Kaplan–Meier estimate; at a tied time, events precede censoring."""
    t = finite(times)
    e = np.asarray(events, dtype=bool)
    if t.size == 0 or t.size != e.size or np.any(t < 0):
        raise ValueError("times must be non-negative, non-empty and match events")
    current, values, risk, counts = 1.0, [], [], []
    distinct = np.unique(t[e])
    for moment in distinct:
        at_risk = int(np.sum(t >= moment))
        deaths = int(np.sum((t == moment) & e))
        current *= 1 - deaths / at_risk
        values.append(current)
        risk.append(at_risk)
        counts.append(deaths)
    return Survival(
        tuple(float(x) for x in distinct),
        tuple(values),
        tuple(risk),
        tuple(counts),
        int(t.size),
        float(t.max()),
    )


def km_quantile(curve: Survival, probability: float) -> float | None:
    """The time by which a share ``probability`` has had the event: the first t with Ŝ(t) ≤ 1 − p, or the
    midpoint of the interval on which Ŝ equals 1 − p exactly, that interval ending at the next event time or,
    without one, at the curve's last observed time (RD-10; the rule of R's ``survival`` package). None when not
    reached."""
    level = 1 - probability
    for index, value in enumerate(curve.survival):
        if value < level - TOLERANCE:
            return curve.times[index]
        if abs(value - level) <= TOLERANCE:
            following = curve.times[index + 1] if index + 1 < len(curve.times) else curve.last_time
            return (curve.times[index] + following) / 2
    return None


def rmst(curve: Survival, horizon: float) -> float:
    """∫₀ʰ Ŝ(t) dt."""
    area, previous_time, previous_value = 0.0, 0.0, 1.0
    for moment, value in zip(curve.times, curve.survival, strict=True):
        if moment >= horizon:
            break
        area += previous_value * (moment - previous_time)
        previous_time, previous_value = moment, value
    return area + previous_value * (horizon - previous_time)


def median_interval(
    restricted: Values, events: Sequence[bool] | NDArray[np.bool_], alpha: float
) -> Interval:
    """Order-statistic interval of the median time with censoring only at L (Brookmeyer and Crowley 1982 in
    that case): [X_(r), X_(n−r+1)] covers the median with probability 1 − 2·P(B ≤ r − 1), B ~ Bin(n, ½); the
    largest r reaching 1 − α, or r = 1 with its level when none does. A limit that falls on a censored run is
    "> L" (``upper`` None, said in the note)."""
    values = finite(restricted)
    flags = np.asarray(events, dtype=bool)
    n = values.size
    if n == 0 or flags.size != n:
        raise ValueError("need matching non-empty values and events")
    censored = values[~flags]
    if censored.size and (
        np.ptp(censored) > TOLERANCE or (flags.any() and censored.min() < values[flags].max())
    ):
        raise ValueError("the order-statistic interval needs censoring only at one common L")
    if n < 2:
        return Interval(None, None, 0.0, "order statistics, Bin(n, 1/2)", "fewer than 2 runs")
    order = np.lexsort((~flags, values))  # by time, events before censored runs at the same time
    x, observed = values[order], flags[order]
    r = 1
    while r + 1 <= n - r and 2 * float(stats.binom.cdf(r, n, 0.5)) <= alpha + TOLERANCE:
        r += 1
    level = 1 - 2 * float(stats.binom.cdf(r - 1, n, 0.5))
    lower = float(x[r - 1]) if observed[r - 1] else None
    upper = float(x[n - r]) if observed[n - r] else None
    notes = []
    if lower is None:
        notes.append("lower limit > L")
    if upper is None:
        notes.append("upper limit > L")
    if level < 1 - alpha - TOLERANCE:
        notes.append(f"{n} runs allow at most {level:.4f}")
    return Interval(
        lower,
        upper,
        level,
        "order statistics, Bin(n, 1/2)",
        "; ".join(notes) or None,
        lower_beyond_limit=lower is None,
        upper_beyond_limit=upper is None,
    )


def t_interval(values: Values, alpha: float) -> Interval:
    """Mean ± t·SD/√n (descriptive for RMST(L), the mean of R)."""
    x = finite(values)
    if x.size < 2:
        return Interval(None, None, 1 - alpha, "t", "fewer than 2 values")
    half = float(stats.t.ppf(1 - alpha / 2, x.size - 1)) * float(x.std(ddof=1)) / math.sqrt(x.size)
    mean = float(x.mean())
    return Interval(mean - half, mean + half, 1 - alpha, "t")


# Paired means: sign flips ----------------------------------------------------------------------------------


def _signed_sums(d: Floats) -> Floats:
    """Σ s_i·Δ_i for all 2ⁿ sign patterns (8 MB at 20 pairs)."""
    sums: Floats = np.zeros(1)
    for value in d:
        sums = np.concatenate([sums + value, sums - value])
    return sums


def sign_flip_p(differences: Values, *, resamples: int, seed: int, shift: float = 0.0) -> float:
    """Two-sided sign-flip test of mean(Δ) = shift: P(|Σ s·(Δ − shift)| ≥ |Σ (Δ − shift)|), exact up to 20 pairs,
    else (1 + #extreme)/(B + 1) from B seeded random sign patterns."""
    d = finite(differences) - shift
    if d.size == 0:
        raise ValueError("no differences")
    observed = abs(float(d.sum()))
    tolerance = TOLERANCE * max(1.0, float(np.abs(d).sum()))
    if d.size <= EXACT_SIGN_FLIP_PAIRS:
        return float(np.mean(np.abs(_signed_sums(d)) >= observed - tolerance))
    rng = random_stream(seed, SIGN_FLIP_STREAM)
    extreme, remaining = 0, resamples
    while remaining:
        batch = min(remaining, 10_000)
        signs = rng.choice((-1.0, 1.0), size=(batch, d.size))
        extreme += int(np.sum(np.abs(signs @ d) >= observed - tolerance))
        remaining -= batch
    return (1 + extreme) / (resamples + 1)


def sign_flip_interval(differences: Values, alpha: float) -> Interval:
    """The θ the exact sign-flip test does not reject (below 10 pairs). The test's outcome changes only at the
    means of subsets of Δ, so the set is read at those points and between them; its limits are subset means.
    With 2·2⁻ⁿ > α no θ is ever rejected, and the interval is given at level 1 − 2·2⁻ⁿ (the most extreme θ
    then tie with the two all-same-sign patterns and are rejected)."""
    d = finite(differences)
    n = d.size
    if n == 0 or n > EXACT_SIGN_FLIP_PAIRS:
        raise ValueError("the exact inversion takes 1 to 20 differences")
    smallest = 2 / 2**n
    used = max(alpha, smallest)
    # the sum and size of every subset of Δ
    totals: Floats = np.zeros(1)
    sizes: Floats = np.zeros(1)
    for value in d:
        totals, sizes = np.concatenate([totals + value, totals]), np.concatenate([sizes + 1, sizes])
    points = np.unique(totals[sizes > 0] / sizes[sizes > 0])
    candidates = np.concatenate([points, (points[:-1] + points[1:]) / 2])
    accepted = [
        float(theta)
        for theta in candidates
        if sign_flip_p(d, resamples=0, seed=0, shift=float(theta)) > used + TOLERANCE
    ]
    level = 1 - used
    note = None if used == alpha else f"{n} pairs allow at most level {level:.4f}"
    if not accepted:
        return Interval(None, None, level, "sign-flip inversion", "no θ accepted")
    lower = max((p for p in points if p <= min(accepted)), default=min(accepted))
    upper = min((p for p in points if p >= max(accepted)), default=max(accepted))
    return Interval(float(lower), float(upper), level, "exact sign-flip inversion", note)


def bootstrap_interval(
    clusters: Sequence[Values],
    statistic: Callable[[Floats], float],
    *,
    resamples: int,
    seed: int,
    alpha: float,
) -> Interval:
    """Percentile interval resampling whole clusters with replacement (RD-5, RD-11); refused below 10
    clusters. The note flags a limit equal to the most extreme resample."""
    groups = [finite(group) for group in clusters if len(group)]
    if len(groups) < BOOTSTRAP_MIN_CLUSTERS:
        raise ValueError(f"a bootstrap needs at least {BOOTSTRAP_MIN_CLUSTERS} clusters (RD-5)")
    rng = random_stream(seed, BOOTSTRAP_STREAM)
    replicates = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        replicates[index] = statistic(np.concatenate([groups[i] for i in chosen]))
    lower, upper = (float(v) for v in np.quantile(replicates, [alpha / 2, 1 - alpha / 2]))
    extreme = lower <= float(replicates.min()) or upper >= float(replicates.max())
    return Interval(
        lower,
        upper,
        1 - alpha,
        f"cluster bootstrap, percentile, B = {resamples}, seed {seed} (stream {BOOTSTRAP_STREAM})",
        "a limit is the most extreme resample" if extreme else None,
    )


def paired_mean(
    differences: Values, *, alpha: float, resamples: int, seed: int
) -> tuple[float, Interval, float]:
    """Mean Δ (ΔRMST for restricted times, RD-3) with its interval (the exact sign-flip inversion below 10
    pairs, a block bootstrap from 10) and the matching sign-flip p-value (RD-13)."""
    d = finite(differences)
    if d.size == 0:
        raise ValueError("no differences")
    if d.size < BOOTSTRAP_MIN_CLUSTERS:
        interval = sign_flip_interval(d, alpha)
    else:
        interval = bootstrap_interval(
            [[value] for value in d],
            lambda x: float(x.mean()),
            resamples=resamples,
            seed=seed,
            alpha=alpha,
        )
    return float(d.mean()), interval, sign_flip_p(d, resamples=resamples, seed=seed)


def geometric_ratio(
    profile: Values, baseline: Values, *, alpha: float, resamples: int, seed: int
) -> tuple[float, Interval, float]:
    """exp(mean(log p − log b)) for positive paired values, with the paired-mean interval and test on the log
    scale transformed back (RD-13)."""
    p, b = finite(profile), finite(baseline)
    if p.size != b.size or np.any(p <= 0) or np.any(b <= 0):
        raise ValueError("need positive paired values")
    estimate, interval, pvalue = paired_mean(
        np.log(p) - np.log(b), alpha=alpha, resamples=resamples, seed=seed
    )
    return (
        math.exp(estimate),
        Interval(
            None if interval.lower is None else math.exp(interval.lower),
            None if interval.upper is None else math.exp(interval.upper),
            interval.level,
            interval.method + ", log scale",
            interval.note,
        ),
        pvalue,
    )


# Paired Hodges–Lehmann and the signed-rank test ------------------------------------------------------------


def walsh_averages(differences: Values) -> Floats:
    """Sorted (Δ_i + Δ_j)/2 for i ≤ j."""
    d = finite(differences)
    i, j = np.triu_indices(d.size)
    return np.sort((d[i] + d[j]) / 2)


def hodges_lehmann(differences: Values) -> float:
    """Median of the Walsh averages."""
    walsh = walsh_averages(differences)
    if walsh.size == 0:
        raise ValueError("no differences")
    return float(np.median(walsh))


def _pratt_ranks(d: Floats) -> tuple[Floats, Floats]:
    """Midranks of |Δ| over all pairs (zeros included, Pratt), then the nonzero Δ and their ranks."""
    # Magnitudes within rounding noise of each other (or of zero) count as tied: sorted, a value joins its
    # neighbour's group when the gap is at most TOLERANCE times the largest magnitude.
    magnitudes = np.abs(d)
    tolerance = TOLERANCE * max(1.0, float(magnitudes.max(initial=0.0)))
    order = np.argsort(magnitudes, kind="mergesort")
    groups = np.empty(d.size)
    groups[order] = np.cumsum(np.concatenate([[0.0], np.diff(magnitudes[order]) > tolerance]))
    zero = magnitudes <= tolerance
    ranks = stats.rankdata(np.where(zero, -1.0, groups), method="average")
    return d[~zero], ranks[~zero]


def signed_rank_p(differences: Values, shift: float = 0.0) -> float:
    """Two-sided Wilcoxon signed-rank p-value of Δ − shift with midranks and Pratt's zeros: up to 40 pairs from
    the exact conditional distribution of T⁺ given the ranks (every sign pattern of the nonzero Δ equally
    likely), above from its normal approximation with the conditional variance Σr²/4 (RD-12)."""
    d = finite(differences) - shift
    if d.size == 0:
        raise ValueError("no differences")
    values, ranks = _pratt_ranks(d)
    if values.size == 0:
        return 1.0
    statistic = float(ranks[values > 0].sum())
    centre = float(ranks.sum()) / 2
    if d.size > EXACT_SIGNED_RANK_PAIRS:
        z = (statistic - centre) / math.sqrt(float((ranks**2).sum()) / 4)
        return min(1.0, 2 * float(stats.norm.sf(abs(z))))
    doubled = np.rint(2 * ranks).astype(np.int64)  # midranks are multiples of 1/2
    counts = np.zeros(int(doubled.sum()) + 1)
    counts[0] = 1.0
    for weight in doubled:
        counts[weight:] = counts[weight:] + counts[: counts.size - weight].copy()
    totals = np.arange(counts.size) / 2
    extreme = np.abs(totals - centre) >= abs(statistic - centre) - TOLERANCE
    return min(1.0, float(counts[extreme].sum() / counts.sum()))


def hodges_lehmann_interval(differences: Values, alpha: float) -> Interval:
    """The θ the signed-rank test of Δ − θ does not reject, from the same distribution as its p-value
    (RD-12). The test changes only at Walsh averages, so the set is read at them and between them; without
    ties this is Tukey's interval of Walsh-average order statistics. When even θ beyond every Δ is not
    rejected, the interval is given at the highest level the pairs allow."""
    d = finite(differences)
    if d.size == 0:
        raise ValueError("no differences")
    walsh = np.unique(walsh_averages(d))
    span = float(walsh[-1] - walsh[0]) or 1.0
    # θ beyond every Δ makes every Δ − θ negative: the least p there is
    smallest = signed_rank_p(d, float(walsh[-1]) + span)
    used = max(alpha, smallest)
    candidates = np.concatenate([walsh, (walsh[:-1] + walsh[1:]) / 2])
    accepted = [float(t) for t in candidates if signed_rank_p(d, float(t)) > used + TOLERANCE]
    level = 1 - used
    method = (
        "inverted signed-rank test, exact conditional (midranks, Pratt)"
        if d.size <= EXACT_SIGNED_RANK_PAIRS
        else "inverted signed-rank test, normal approximation (midranks, Pratt)"
    )
    note = None if used == alpha else f"{d.size} pairs allow at most level {level:.4f}"
    if not accepted:
        return Interval(None, None, level, method, "no θ accepted")
    lower = max((w for w in walsh if w <= min(accepted)), default=min(accepted))
    upper = min((w for w in walsh if w >= max(accepted)), default=max(accepted))
    return Interval(float(lower), float(upper), level, method, note)


def paired_hodges_lehmann(differences: Values, alpha: float) -> tuple[float, Interval, float]:
    """Hodges–Lehmann estimate, its interval and the matching signed-rank p-value (RD-13)."""
    return (
        hodges_lehmann(differences),
        hodges_lehmann_interval(differences, alpha),
        signed_rank_p(differences),
    )


# Descriptive -----------------------------------------------------------------------------------------------


def quartiles(values: Values) -> tuple[float, float, float]:
    """Q1, median, Q3 as the averaged inverse empirical distribution (Hyndman and Fan type 2): for complete
    data the Kaplan–Meier midpoint rule of RD-10 gives the same values, so censored and uncensored metrics
    share one quantile rule."""
    x = finite(values)
    if x.size == 0:
        raise ValueError("no values")
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75], method="averaged_inverted_cdf")
    return float(q1), float(q2), float(q3)
