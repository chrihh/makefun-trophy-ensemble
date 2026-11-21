"""GTA VI trophy earn-rate forecaster, v3: a rehearsal-weighted ensemble.

Written for the make.fun bounty gta6-trophy-rarity-forecast-01.

No single story about a launch's first week is trustworthy, so this
program tells itself three different stories about every trophy and then
lets the trophy page's own history judge which story to believe:

  * the cap story -- the rate bends toward a ceiling, fitted as
    rate(u) = cap * (1 - exp(-speed * u ** 1.2)) on log-time since
    release, the speed grid-searched and shrunk toward the field's
    typical speed, the cap solved by least squares;
  * the ratio story -- an earn rate is earners over tracked players, and
    the two sides are extrapolated apart: earners by a damped log-time
    growth fit shrunk toward the pooled growth of all trophies, players
    by a damped log-time fit of the pool itself, then divided;
  * the trend story -- the rate's logit simply continues its recent
    log-time slope, shrunk toward the pooled slope and damped.

The judging is a rehearsal: the program hides the last distinct page
state, refits all three stories one refresh early, and scores each
story's predictions against the state it hid. Stories that rehearsed
well earn weight in the final blend, in proportion to the inverse square
of their rehearsal error, with a floor under each so no story is ever
silenced entirely -- the last refresh is one data point, not a verdict.
The final median is the weighted blend of the three stories' logits.

The spread is built from the stories' disagreement with each other, the
cap story's fit residuals, the counting noise implied by each trophy's
earner count, and the ensemble's own rehearsal error, which also sets a
small global bias correction.

Page states are deduplicated first: Exophase refreshes in jumps, and a
read that repeats the previous one adds no information. With fewer than
three distinct states no story can be fitted or judged, and the program
holds the last rate inside a wide band that prices in an unseen refresh.

Deterministic, standard library only, no network, no clock reading.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
ZVALS = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
POWER = 1.2
SPEEDS = [0.05 * (1.20 ** i) for i in range(36)]


def to_epoch(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def days_since(epoch):
    return max((epoch - LAUNCH) / 86400.0, 0.0)


def u_of(days):
    return math.log1p(days)


def logit_of(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def middle(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def slope_of(xs, ys, ws):
    """Weighted least-squares slope, or 0.0 when undefined."""
    total = sum(ws)
    if total <= 0 or len(xs) < 2:
        return 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / total
    my = sum(w * y for y, w in zip(ys, ws)) / total
    var = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if var < 1e-12:
        return 0.0
    cov = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    return cov / var


def page_states(case):
    """History collapsed to the reads where the printed page changed."""
    states = []
    for read in case["history"]:
        marker = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1]["marker"] == marker:
            continue
        states.append(
            {
                "marker": marker,
                "days": days_since(to_epoch(read.get("slot") or read["at"])),
                "players": float(read["players_tracked"]),
                "rates": read["rates"],
            }
        )
    return states


def series_of(states, idx):
    """(u, rate, players) points for one trophy across the states."""
    pts = []
    for s in states:
        if idx < len(s["rates"]) and s["rates"][idx] is not None:
            pts.append((u_of(s["days"]), float(s["rates"][idx]), s["players"]))
    return pts


# ---------------------------------------------------------------- cap story

def fit_cap(us, rates):
    """Grid-search the saturating curve; return (speed, cap, rmse)."""
    best = None
    for speed in SPEEDS:
        basis = [1.0 - math.exp(-speed * (u ** POWER)) for u in us]
        denom = sum(b * b for b in basis)
        if denom < 1e-12:
            continue
        cap = sum(r * b for r, b in zip(rates, basis)) / denom
        if cap <= 0:
            continue
        cap = min(cap, 100.0)
        sse = 0.0
        for r, b in zip(rates, basis):
            sse += (logit_of(r) - logit_of(max(cap * b, 0.01))) ** 2
        if best is None or sse < best[0]:
            best = (sse, speed, cap)
    if best is None:
        return None
    sse, speed, cap = best
    return speed, cap, math.sqrt(sse / max(len(us), 1))


def cap_story(states, target_days):
    """Predicted rate per trophy from the saturating curve, plus rmse."""
    u_tgt = u_of(target_days)
    n = len(states[0]["rates"])
    raw = []
    for idx in range(n):
        pts = series_of(states, idx)
        if len(pts) < 3:
            raw.append(None)
            continue
        got = fit_cap([p[0] for p in pts], [p[1] for p in pts])
        if got is None:
            raw.append(None)
            continue
        speed, cap, rmse = got
        raw.append({"speed": speed, "cap": cap, "rmse": rmse, "pts": pts,
                    "r_last": pts[-1][1], "e_last": pts[-1][1] * pts[-1][2] / 100.0})
    solid = [f["speed"] for f in raw if f and f["e_last"] >= 20.0]
    typical = middle(solid) if solid else (middle([f["speed"] for f in raw if f]) if any(raw) else 0.5)
    out = []
    for f in raw:
        if f is None:
            out.append(None)
            continue
        trust = f["e_last"] / (f["e_last"] + 45.0)
        speed = math.exp(trust * math.log(f["speed"]) + (1.0 - trust) * math.log(typical))
        us = [p[0] for p in f["pts"]]
        rs = [p[1] for p in f["pts"]]
        basis = [1.0 - math.exp(-speed * (u ** POWER)) for u in us]
        denom = sum(b * b for b in basis)
        cap = sum(r * b for r, b in zip(rs, basis)) / denom if denom > 1e-12 else f["cap"]
        cap = min(max(cap, f["r_last"]), 100.0)
        pred = cap * (1.0 - math.exp(-speed * (u_tgt ** POWER)))
        pred = min(max(pred, 0.30 * f["r_last"]), 99.5)
        out.append({"pred": pred, "rmse": f["rmse"]})
    return out


# -------------------------------------------------------------- ratio story

def pool_target(states, target_days):
    """Tracked players at the target from a damped log-time fit."""
    last = states[-1]
    p_last = max(last["players"], 1.0)
    if len(states) < 2:
        return p_last
    xs = [u_of(s["days"]) for s in states]
    ys = [math.log(max(s["players"], 1.0)) for s in states]
    ws = [0.5 ** ((last["days"] - s["days"]) / 3.0) for s in states]
    slope = max(0.0, slope_of(xs, ys, ws))
    grown = math.log(p_last) + 0.8 * slope * (u_of(target_days) - xs[-1])
    grown = min(grown, math.log(p_last) + 1.2 + 0.25 * max(target_days - last["days"], 0.0))
    return max(math.exp(grown), p_last)


def ratio_story(states, target_days):
    """Predicted rate per trophy from earners and players forecast apart."""
    n = len(states[0]["rates"])
    pool_t = pool_target(states, target_days)
    u_tgt = u_of(target_days)
    last_days = states[-1]["days"]
    ws_all = [0.5 ** ((last_days - s["days"]) / 3.0) for s in states]
    raw = []
    for idx in range(n):
        pts = series_of(states, idx)
        if len(pts) < 3:
            raw.append(None)
            continue
        us = [p[0] for p in pts]
        earn = [p[1] * p[2] / 100.0 for p in pts]
        wts = [ws_all[min(i, len(ws_all) - 1)] for i in range(len(pts))]
        # Align weights by matching each point's u back to its state.
        pos = {u_of(s["days"]): w for s, w in zip(states, ws_all)}
        wts = [pos.get(u, 0.3) for u in us]
        slope_e = slope_of(us, [math.log(e + 1.0) for e in earn], wts)
        raw.append({"slope_e": slope_e, "e_last": earn[-1], "x_last": us[-1],
                    "r_last": pts[-1][1]})
    solid = [f["slope_e"] for f in raw if f and f["e_last"] >= 15.0]
    pooled = middle(solid) if solid else (middle([f["slope_e"] for f in raw if f]) if any(raw) else 0.0)
    out = []
    for f in raw:
        if f is None:
            out.append(None)
            continue
        trust = f["e_last"] / (f["e_last"] + 25.0)
        slope = trust * f["slope_e"] + (1.0 - trust) * pooled
        slope = min(max(slope, -0.5), 3.0)
        e_pred = math.exp(math.log(f["e_last"] + 1.0) + 0.75 * slope * (u_tgt - f["x_last"])) - 1.0
        e_pred = min(max(e_pred, f["e_last"]), 0.999 * pool_t)
        out.append({"pred": 100.0 * e_pred / max(pool_t, 1.0)})
    return out


# -------------------------------------------------------------- trend story

def trend_story(states, target_days):
    """Predicted rate per trophy from a shrunk, damped logit trend."""
    n = len(states[0]["rates"])
    u_tgt = u_of(target_days)
    last_days = states[-1]["days"]
    ws_all = [0.5 ** ((last_days - s["days"]) / 3.0) for s in states]
    pos = {u_of(s["days"]): w for s, w in zip(states, ws_all)}
    raw = []
    for idx in range(n):
        pts = series_of(states, idx)
        if len(pts) < 3:
            raw.append(None)
            continue
        us = [p[0] for p in pts]
        wts = [pos.get(u, 0.3) for u in us]
        slope_l = slope_of(us, [logit_of(p[1]) for p in pts], wts)
        raw.append({"slope_l": slope_l, "r_last": pts[-1][1], "x_last": us[-1],
                    "e_last": pts[-1][1] * pts[-1][2] / 100.0})
    solid = [f["slope_l"] for f in raw if f and f["e_last"] >= 15.0]
    pooled = middle(solid) if solid else (middle([f["slope_l"] for f in raw if f]) if any(raw) else 0.0)
    out = []
    for f in raw:
        if f is None:
            out.append(None)
            continue
        trust = f["e_last"] / (f["e_last"] + 25.0)
        slope = trust * f["slope_l"] + (1.0 - trust) * pooled
        slope = min(max(slope, -1.0), 3.0)
        lg = logit_of(f["r_last"]) + 0.7 * slope * (u_tgt - f["x_last"])
        out.append({"pred": pct_of_logit(lg)})
    return out


STORIES = (cap_story, ratio_story, trend_story)


def run_stories(states, target_days):
    return [story(states, target_days) for story in STORIES]


def judge_stories(states):
    """Rehearse each story one refresh early; return story weights, bias
    and a band widening for the ensemble as a whole."""
    default = ([1.0 / 3.0] * 3, 0.0, 1.0)
    if len(states) < 5:
        return default
    head, truth = states[:-1], states[-1]
    rehearsed = run_stories(head, truth["days"])
    errors = [[], [], []]
    blended_err = []
    for idx in range(len(truth["rates"])):
        if truth["rates"][idx] is None:
            continue
        actual = logit_of(float(truth["rates"][idx]))
        preds = []
        for s in range(3):
            f = rehearsed[s][idx] if idx < len(rehearsed[s]) else None
            if f is None:
                preds.append(None)
            else:
                preds.append(logit_of(f["pred"]))
                errors[s].append(abs(actual - preds[s]))
        if all(p is not None for p in preds):
            blended_err.append(actual - sum(preds) / 3.0)
    med_err = []
    for s in range(3):
        if len(errors[s]) < 12:
            return default
        med_err.append(middle(errors[s]))
    # A prior on top of the rehearsal: the cap story is the only one whose
    # shape matches how trophy progress actually accumulates, and a single
    # rehearsed refresh is a short-horizon test that flatters the trend
    # stories, so the cap story's rehearsal score counts for more.
    prior = [1.7, 1.0, 1.0]
    raw_w = [p / (e + 0.15) ** 2 for p, e in zip(prior, med_err)]
    total = sum(raw_w)
    weights = [w / total for w in raw_w]
    # Floors and ceilings: no story silenced, no story supreme on the
    # evidence of a single refresh.
    floors = [0.38, 0.10, 0.10]
    ceilings = [0.82, 0.60, 0.60]
    weights = [min(max(w, lo), hi) for w, lo, hi in zip(weights, floors, ceilings)]
    total = sum(weights)
    weights = [w / total for w in weights]
    if len(blended_err) >= 12:
        bias = min(max(0.5 * middle(blended_err), -0.28), 0.28)
        widen = min(max(middle([abs(e) for e in blended_err]) / 0.45, 0.75), 1.9)
    else:
        bias, widen = 0.0, 1.0
    return weights, bias, widen


def tidy4(value):
    rounded = round(float(value) + 1e-12, 4)
    return int(rounded) if float(rounded).is_integer() else rounded


def forecast_one(case):
    if case["levels"] != QUANTILES:
        raise ValueError("unexpected quantile levels")
    states = page_states(case)
    if not states:
        raise ValueError("empty history")
    target_days = days_since(to_epoch(case["target"]))
    last = states[-1]
    horizon_days = max(target_days - last["days"], 0.0)

    if len(states) < 3:
        sigma = 0.70 * math.sqrt(horizon_days) + 0.55
        rows = []
        for idx, trophy in enumerate(case["trophies"]):
            rate = last["rates"][idx] if idx < len(last["rates"]) else None
            centre = logit_of(float(rate)) if rate is not None else logit_of(0.4)
            qs = [tidy4(pct_of_logit(centre + z * sigma)) for z in ZVALS]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    stories = run_stories(states, target_days)
    weights, bias, widen = judge_stories(states)
    caps = stories[0]

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate_last = last["rates"][idx] if idx < len(last["rates"]) else None
        r_last = float(rate_last) if rate_last is not None else 0.4
        e_last = r_last * last["players"] / 100.0
        logits, used_w = [], []
        for s in range(3):
            f = stories[s][idx] if idx < len(stories[s]) else None
            if f is not None:
                logits.append(logit_of(f["pred"]))
                used_w.append(weights[s])
        if not logits:
            centre = logit_of(r_last)
            sigma = 0.9
        else:
            wsum = sum(used_w)
            centre = sum(w * lg for w, lg in zip(used_w, logits)) / wsum + bias
            spread = math.sqrt(
                sum(w * (lg - centre) ** 2 for w, lg in zip(used_w, logits)) / wsum
            )
            rmse = caps[idx]["rmse"] if caps[idx] else 0.35
            counting = 1.05 / math.sqrt(e_last + 3.0)
            sigma = math.sqrt(
                (0.24 * math.sqrt(horizon_days)) ** 2
                + (0.80 * rmse) ** 2
                + counting ** 2
                + (0.55 * spread) ** 2
                + 0.09 ** 2
            )
            sigma = min(max(sigma * widen, 0.32), 1.65)
        qs = [tidy4(pct_of_logit(centre + z * sigma)) for z in ZVALS]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_one(case)
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    run()
