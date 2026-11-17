"""gta6-twitch-ccu-forecast-01 entry: local-linear log-ratio forecaster.

Model
-----
Work in log-ratios to the hour-1 print. For every comparable past launch,
take its release-day (UTC) hourly viewers v_1..v_24 and form
r_h = ln(v_h / v_1) for h = 2..24. The key empirical fact in the practice
data is that r_h depends strongly on how big hour 1 already was: for the
biggest launches (hour-1 viewers in the hundreds of thousands) the day
stays near hour 1 (ratios roughly 0.4-1.3 all day), while for small
categories hour 1 is still pre-unlock quiet and the day explodes to
100x and more later (corr(log v_1, r_h) is about -0.9 by the evening).
So for each hour separately we fit a *local* linear regression of r_h on
ln(v_1), kernel-weighted (Gaussian, bandwidth 1.75 in log units) around
this case's own ln(hour-1 print): near launches of the same size matter
most, and at GTA VI scale the fit leans on the biggest launches only,
which is also what keeps the extrapolation sane. The forecast median is
p1 * exp(mu_h); the spread is the kernel-weighted residual standard
deviation (floored at 0.20 log units), shrunk by 0.72, applied through
the standard-normal quantiles at the five scored levels. The 0.72 was
chosen on the practice cases: it lands 90% coverage near 0.95 with the
lowest mean pinball loss among the spread values tried.

Why this shape, in words: hour 1 (00:00-01:00 UTC) is pre-unlock almost
everywhere except New Zealand, so the day dips to a trough around hours
3-5, steps up at hours 5-6 as the US East Coast unlocks and wakes, and
peaks in the evening (hours 16-21) when the US is playing and Europe is
watching. The per-hour fits recover that shape from the comparables
themselves; nothing about the unlock clock is hardcoded.

Practice result (local scoring against the SullyGnome hourly averages
that the practice answers come from; the kit ships no local grader):
mean pinball loss 0.176, typical miss 31.6%, 90% coverage 0.957,
against the house persistence baseline's 0.387 / 82.7% / 0.783.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Gaussian kernel bandwidth (log-viewer units) for the local fits.
BANDWIDTH = 1.75
# Shrinkage applied to the residual spread; tuned on the practice cases.
SPREAD_SHRINK = 0.72
# Floor on the per-hour residual log-spread, so no hour claims false certainty.
SD_FLOOR = 0.20
# Fallback spread (log units) if the comparables give us nothing usable.
FALLBACK_SIGMA = 0.15


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


def comparable_curves(case):
    """Per comparable: (ln of release-day hour-1 viewers, {hour: log-ratio}).

    Only the comparable's own UTC release day is used, hours 2..24
    divided by hour 1. A comparable with no usable hour-1 print (the
    GTA VI category's own pre-release history is one: it has no release
    day yet) contributes nothing.
    """
    curves = []
    for comp in case.get("comparables") or []:
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
        if ratios:
            curves.append((math.log(first), ratios))
    return curves


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


def hour_quantiles(p1, log_p1, curves, hour):
    """The five forecast quantiles for one hour, as viewer counts."""
    points = [(lx, ratios[hour]) for lx, ratios in curves if hour in ratios]
    if len(points) >= 3:
        weights = [math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2)
                   for x, _y in points]
        weight_sum = sum(weights)
        if weight_sum < 1.0:
            # This case sits far outside every comparable's size (an
            # hour-1 print far above the biggest launch on record): the
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
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        hours.append({"hour": hour,
                      "q": clamp_quantiles(hour_quantiles(p1, log_p1, curves,
                                                           hour))})
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
