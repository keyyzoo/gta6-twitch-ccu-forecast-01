# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/keyyzoo/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/keyyzoo/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles per hour, from the first hour's viewer count and the release-day curves of past big game launches. The model works in log-ratios to the hour-1 print and fits, separately for each hour, a local linear regression of the log-ratio on log hour-1 size, kernel-weighted around this case's own size — because the day's shape depends strongly on how big hour 1 already was: the biggest launches stay near hour 1 all day, while small categories explode 100x or more by the evening. v2 adds one feature to that fit: the share of the game's pre-release viewers in the last 48h before release day who watched in 00:00–06:00 UTC, the overnight share of the pre-release audience — a timezone-mix signal that matters most in the evening peak hours. On the practice cases it scores a mean pinball loss of 0.154 (typical miss 27.6%, 90% coverage 0.935), against v1's 0.176 and the house persistence baseline's 0.387.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=mif&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
