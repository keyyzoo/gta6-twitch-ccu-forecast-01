"""gta6-twitch-ccu-forecast-01 entry v2: local-linear log-ratio forecaster
with a pre-release diurnal-phase feature.

Model
-----
v1 (loss 0.1758) fit, per hour, a kernel-weighted local-linear regression
of the log-ratio r_h = ln(v_h / v_1) on ln(hour-1 viewers) across past
launches. v2 adds one feature: the share of a game's pre-release
viewers in the last 48h before release day who watched in 00:00-06:00
UTC ("s00"), the overnight share of the pre-release audience. It
measures the audience's timezone mix: a launch whose pre-release
audience is already up overnight behaves differently through the day
than one whose audience only wakes for the evening peak. The feature
enters linearly (partial-linear model): the kernel weights depend on
size only, and the fit is r_h ~ 1 + ln(v_1) + s00, centered at the
case's own values, so the intercept is the forecast log-ratio for the
case. Spread is the kernel-weighted residual SD (floored at 0.20 log
units), shrunk by 0.65.

Why s00, and why 48h: on the four practice cases the feature cuts the
mean pinball loss from 0.1758 (v1) to 0.1538, improving three of the
four cases (p00 0.169 -> 0.158, p01 0.207 -> 0.158, p02 0.145 -> 0.112;
p03 0.183 -> 0.188), with the gains concentrated in the evening peak
hours (17-21) where v1 missed most. A 48h window matches what the
final case carries (the house recorder started 2026-11-17, so h00's
history spans ~43h); the 7-day window scores worse (0.1615). The
feature is computed identically for the case and every comparable,
from the hourly rows in the 48h before each release day.

Fallbacks (all deterministic, no network):
- No usable pre-release history for the case, or fewer than 8
  comparables carrying the feature in some hour: the whole case falls
  back to v1's size-only local-linear fit.
- A comparable whose name matches the case's own game is skipped: in
  the final the comparables include the GTA VI category's own
  pre-release history, which is not a launch day and must not pose as
  one. (No practice case has such a match, so practice scores are
  unaffected.)
- Too few curves for an hour: pooled empirical quantiles; no curves at
  all: the house baseline's widening spread around hour 1.

Practice result (local scoring against the SullyGnome hourly averages
that the practice answers come from; the kit ships no local grader):
mean pinball loss 0.1538, typical miss 27.6%, 90% coverage 0.935,
against v1's 0.1758 / 31.6% / 0.957 and the house persistence
baseline's 0.3871 / 82.7% / 0.783.

Run as the grader runs it: `python3 -P -s /bot/main.py` (Python 3.13,
standard library only, no network). Reads every <id>.in in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"hours": [{"hour": h,
"q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24. Deterministic:
no randomness anywhere, so two runs write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Gaussian kernel bandwidth (log-viewer units) for the local fits.
BANDWIDTH = 1.5
# Shrinkage applied to the residual spread; tuned on the practice cases.
SPREAD_SHRINK = 0.65
# Floor on the per-hour residual log-spread, so no hour claims false certainty.
SD_FLOOR = 0.20
# Fallback spread (log units) if the comparables give us nothing usable.
FALLBACK_SIGMA = 0.15
# Pre-release window (days) for the diurnal feature.
DIURNAL_DAYS = 2
# Minimum comparables carrying the feature for the partial-linear fit.
MIN_FEATURE_CURVES = 8
# Minimum hourly history rows for a usable diurnal feature.
MIN_HISTORY_ROWS = 12


def parse_time(text):
    """ISO 8601 timestamp -> aware datetime (UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, as JavaScript's Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def clamp_quantiles(values):
    """Make five numbers finite, > 0, <= 1e9, non-decreasing, 2-decimal."""
    out = []
    prev = 0.0
    for value in values:
        if not math.isfinite(value):
            value = prev if prev > 0 else 1.0
        value = min(max(value, 1.0), 1e9)
        value = max(value, prev)
        out.append(round2(value))
        prev = out[-1]
    return out


def solve_linear(matrix, rhs):
    """Small dense linear solve by Gaussian elimination; None if singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                for cc in range(col, n + 1):
                    rows[r][cc] -= factor * rows[col][cc]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def wls_multi(features, values, weights):
    """Weighted least squares with intercept; returns coefficient list."""
    width = len(features[0]) + 1
    matrix = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for xs, y, w in zip(features, values, weights):
        row = [1.0] + list(xs)
        for i in range(width):
            rhs[i] += w * row[i] * y
            for j in range(width):
                matrix[i][j] += w * row[i] * row[j]
    return solve_linear(matrix, rhs)


def weighted_line(points, weights):
    """Weighted least squares of y on x; returns (intercept, slope)."""
    total = sum(weights)
    if total <= 0:
        return 0.0, 0.0
    mean_x = sum(w * x for w, (x, _y) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_x, y) in zip(weights, points)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, (x, _y) in zip(weights, points))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, (x, y) in zip(weights, points))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    return mean_y - slope * mean_x, slope


def quantile_of(sorted_values, level):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = level * (n - 1)
    low = int(pos)
    high = min(low + 1, n - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def diurnal_s00(rows, day):
    """Share of pre-release viewers in 00:00-06:00 UTC over the last
    DIURNAL_DAYS days before `day`. None when the history is too thin
    to say anything (too few rows or no viewers at all)."""
    totals = [0.0] * 24
    count = 0
    start = day - timedelta(days=DIURNAL_DAYS)
    for row in rows or []:
        try:
            text = row.get("from") or row.get("at")
            at = parse_time(text)
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if not isinstance(viewers, (int, float)) or viewers is None:
            continue
        if start <= at < day:
            totals[at.hour] += max(viewers, 0.0)
            count += 1
    total = sum(totals)
    if total <= 0 or count < MIN_HISTORY_ROWS:
        return None
    return sum(totals[0:6]) / total


def comparable_curves(case):
    """Per comparable: (ln of release-day hour-1 viewers, {hour: log-ratio},
    overnight pre-release share). Only the comparable's own UTC release
    day is used, hours 2..24 divided by hour 1. A comparable with no
    usable hour-1 print contributes nothing. A comparable whose name
    matches the case's own game is skipped: in the final that is the
    GTA VI category's own pre-release history, not a launch day."""
    game = case.get("game") or {}
    game_name = (game.get("name") or "").strip().lower() \
        if isinstance(game, dict) else ""
    curves = []
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or "").strip().lower()
        if game_name and name and name == game_name:
            continue
        try:
            day = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hourly = {}
        for row in comp.get("hourly") or []:
            try:
                at = parse_time(row["from"])
                viewers = row["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(viewers, (int, float)) or viewers <= 0:
                continue
            if day <= at < day + timedelta(days=1):
                hourly[(at - day).seconds // 3600 + 1] = float(viewers)
        if 1 not in hourly:
            continue
        first = hourly[1]
        ratios = {h: math.log(v / first) for h, v in hourly.items() if h >= 2}
        if not ratios:
            continue
        curves.append({
            "logv1": math.log(first),
            "ratios": ratios,
            "s00": diurnal_s00(comp.get("hourly") or [], day),
        })
    return curves


def case_s00(case):
    """The case's own overnight pre-release share, from its history."""
    try:
        day = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    rows = case.get("history_hourly") or case.get("history") or []
    return diurnal_s00(rows, day)


def hour_quantiles_partial(p1, log_p1, s00_case, curves, hour):
    """The five forecast quantiles for one hour, partial-linear fit:
    kernel on size, overnight share entered linearly, centered at the
    case's own values so the intercept is the forecast log-ratio."""
    pts = [c for c in curves if hour in c["ratios"] and c["s00"] is not None]
    features = [[c["logv1"] - log_p1, c["s00"] - s00_case] for c in pts]
    values = [c["ratios"][hour] for c in pts]
    weights = [math.exp(-0.5 * ((c["logv1"] - log_p1) / BANDWIDTH) ** 2)
               for c in pts]
    beta = wls_multi(features, values, weights)
    if beta is None:
        beta = [sum(w * y for w, y in zip(weights, values)) / sum(weights),
                0.0, 0.0]
    mu = beta[0]
    resid = [y - (beta[0] + beta[1] * xs[0] + beta[2] * xs[1])
             for xs, y in zip(features, values)]
    weight_sum = sum(weights)
    spread = math.sqrt(
        sum(w * r * r for w, r in zip(weights, resid))
        / max(1.0, weight_sum - 4.0))
    spread = max(spread, SD_FLOOR) * SPREAD_SHRINK
    return [p1 * math.exp(mu + z * spread) for z in Z]


def hour_quantiles_size_only(p1, log_p1, curves, hour):
    """v1's size-only local-linear fit, used when the diurnal feature is
    unavailable for the case as a whole."""
    points = [(c["logv1"], c["ratios"][hour]) for c in curves
              if hour in c["ratios"]]
    if len(points) >= 3:
        weights = [math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2)
                   for x, _y in points]
        weight_sum = sum(weights)
        if weight_sum < 1.0:
            # This case sits far outside every comparable's size: the
            # kernel has almost no mass, so a local fit would be noise.
            # Use the pooled empirical quantiles instead.
            values = sorted(y for _x, y in points)
            return [p1 * math.exp(quantile_of(values, level))
                    for level in LEVELS]
        intercept, slope = weighted_line(points, weights)
        centre = intercept + slope * log_p1
        spread = math.sqrt(
            sum(w * (y - (intercept + slope * x)) ** 2
                for w, (x, y) in zip(weights, points))
            / max(1.0, weight_sum - 2.0))
        spread = max(spread, SD_FLOOR) * SPREAD_SHRINK
        return [p1 * math.exp(centre + z * spread) for z in Z]
    if points:
        # Too few curves for a local fit: fall back to their pooled
        # empirical quantiles for this hour.
        values = sorted(y for _x, y in points)
        return [p1 * math.exp(quantile_of(values, level)) for level in LEVELS]
    # No comparable reports this hour at all: stay near hour 1 with a
    # spread that widens slowly over the day, as the house baseline does.
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]


def forecast_case(case, name):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    curves = comparable_curves(case)
    s00_case = case_s00(case)
    use_feature = s00_case is not None
    if use_feature:
        # The partial fit needs enough feature-carrying curves in every
        # hour; otherwise the whole case falls back to size-only.
        for row in case.get("forecast") or []:
            hour = row["hour"]
            n = sum(1 for c in curves
                    if hour in c["ratios"] and c["s00"] is not None)
            if n < MIN_FEATURE_CURVES:
                use_feature = False
                break
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        if use_feature:
            qs = hour_quantiles_partial(p1, log_p1, s00_case, curves, hour)
        else:
            qs = hour_quantiles_size_only(p1, log_p1, curves, hour)
        hours.append({"hour": hour, "q": clamp_quantiles(qs)})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
