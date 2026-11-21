# makefun-trophy-ensemble

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chrihh/makefun-trophy-ensemble/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chrihh/makefun-trophy-ensemble/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the make.fun bounty "Forecast how many GTA VI players earn
each trophy in week one" that refuses to bet on a single story. `main.py`
runs three models per trophy — a saturating cap curve on log-time
(`rate = cap * (1 - exp(-speed * u ** 1.2))`), an earners-over-players
ratio model that extrapolates the two sides of the rate apart, and a
shrunk, damped logit trend — then judges them against each other in a
rehearsal: it hides the last distinct Exophase page state, refits all
three one refresh early, and weights the final blend by each story's
rehearsal error, with a prior in favour of the cap curve and floors so no
story is silenced by one refresh. Bands come from the stories'
disagreement, the cap fit's residuals, counting noise and the ensemble's
own rehearsal error. On synthetic launch worlds built for testing
(saturating, dilution-heavy, flat and logit-trend shapes) it scored
pinball losses of roughly 0.11–0.25 against the persistence baseline's
0.23–0.41 on the saturating worlds, and stayed ahead of the baseline on
every other shape too. Standard library only, deterministic, well under
a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds one or more `<id>.in` case files in the contest's
`makefun/gta6-trophies-case/v1` format; one `<id>.out` forecast is written
per case, each with five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

---

Entered on [make.fun](https://make.fun/?ref=44n6pkpv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
