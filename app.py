import time
import math
import io
import os
import tempfile
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


# =============================================================================
# Page configuration and global constants
# =============================================================================

st.set_page_config(
    page_title="Air-Traffic Conflict Resolution Simulator",
    page_icon="✈️",
    layout="wide",
)

APP_TITLE = "Research-Grade Air-Traffic Conflict-Resolution Simulator"

DISCLAIMER = (
    "SIMULATION ONLY. This is a synthetic research/training environment. "
    "It is NOT operational aviation software and must not be used for real-world "
    "navigation, air-traffic control, or safety-critical decision-making."
)

# Default separation constraints
H_SEP_KM_DEFAULT = 9.26       # 5 NM in km
V_SEP_M_DEFAULT = 305.0       # 1000 ft in m

# Prediction and planning parameters
PRED_DT = 5.0                 # seconds per prediction step
LOOKAHEAD = 140.0             # forecast horizon in simulated seconds
REPLAN_COOLDOWN = 25.0        # simulated seconds between automatic replans

# Demonstration timing
DEMO_REAL_SECONDS = 80.0      # requested live demonstration duration
TOTAL_DEMO_SIM_SECONDS = 320.0  # accelerated simulated scenario length for recorder

CONFLICT_RISK_THRESHOLD = 0.20


# =============================================================================
# Core data model
# =============================================================================

@dataclass
class Aircraft:
    id: str
    callsign: str
    x: float                  # km east
    y: float                  # km north
    alt: float                # m
    heading: float            # radians, 0 = north, pi/2 = east
    speed: float              # km/s
    target_heading: float
    target_speed: float
    target_alt: float
    nominal_heading: float
    nominal_speed: float
    nominal_alt: float
    origin: str
    dest: str
    color: str
    uncertainty_km: float = 0.35
    status: str = "NOMINAL"
    maneuver: str = ""
    maneuver_start: Optional[float] = None


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rad_to_deg(rad: float) -> float:
    return (rad * 180.0 / math.pi) % 360.0


def init_histories(world: Dict[str, Aircraft]) -> Dict[str, dict]:
    histories = {}
    for ac_id, ac in world.items():
        histories[ac_id] = {
            "t": [0.0],
            "x": [ac.x],
            "y": [ac.y],
            "z": [ac.alt],
        }
    return histories


# =============================================================================
# Demo scenario generation
# =============================================================================

def create_demo_world(seed: int = 7) -> Dict[str, Aircraft]:
    random.seed(seed)
    np.random.seed(seed)

    world: Dict[str, Aircraft] = {}

    def add_ac(
        ac_id: str,
        callsign: str,
        x: float,
        y: float,
        alt: float,
        heading: float,
        speed: float,
        origin: str,
        dest: str,
        color: str,
    ):
        world[ac_id] = Aircraft(
            id=ac_id,
            callsign=callsign,
            x=x,
            y=y,
            alt=alt,
            heading=heading,
            speed=speed,
            target_heading=heading,
            target_speed=speed,
            target_alt=alt,
            nominal_heading=heading,
            nominal_speed=speed,
            nominal_alt=alt,
            origin=origin,
            dest=dest,
            color=color,
            uncertainty_km=0.35,
            status="NOMINAL",
            maneuver="",
            maneuver_start=None,
        )

    # Primary crossing conflict around simulated t = 150 s
    add_ac(
        "AAL10",
        "AAL10",
        x=-35.0,
        y=0.0,
        alt=10000.0,
        heading=math.pi / 2.0,
        speed=0.230,
        origin="KSFO",
        dest="KJFK",
        color="#e74c3c",
    )

    add_ac(
        "UAL45",
        "UAL45",
        x=0.0,
        y=-35.0,
        alt=10000.0,
        heading=0.0,
        speed=0.230,
        origin="KATL",
        dest="KORD",
        color="#3498db",
    )

    # Secondary crossing conflict around simulated t = 260 s
    add_ac(
        "DAL63",
        "DAL63",
        x=-55.0,
        y=15.0,
        alt=10100.0,
        heading=math.pi / 2.0,
        speed=0.228,
        origin="KDEN",
        dest="KBOS",
        color="#2ecc71",
    )

    add_ac(
        "SWA88",
        "SWA88",
        x=5.0,
        y=75.0,
        alt=10100.0,
        heading=math.pi,
        speed=0.228,
        origin="KMDW",
        dest="KAUS",
        color="#f1c40f",
    )

    # Non-conflicting traffic with vertical separation
    add_ac(
        "AFR090",
        "AFR090",
        x=-40.0,
        y=-40.0,
        alt=9400.0,
        heading=math.radians(45.0),
        speed=0.220,
        origin="KORD",
        dest="KLGA",
        color="#9b59b6",
    )

    add_ac(
        "BAW216",
        "BAW216",
        x=40.0,
        y=40.0,
        alt=10650.0,
        heading=math.radians(225.0),
        speed=0.220,
        origin="EGLL",
        dest="KJFK",
        color="#1abc9c",
    )

    return world


# =============================================================================
# Aircraft dynamics, forecasting, and conflict prediction
# =============================================================================

def step_aircraft(ac: Aircraft, dt: float) -> None:
    if dt <= 0.0:
        return

    # Heading dynamics: limited turn rate
    max_turn = math.radians(3.0) * dt
    dh = normalize_angle(ac.target_heading - ac.heading)
    if abs(dh) > max_turn:
        dh = math.copysign(max_turn, dh)
    ac.heading = normalize_angle(ac.heading + dh)

    # Speed dynamics: limited acceleration
    max_accel = 0.004 * dt  # km/s per step
    ds = ac.target_speed - ac.speed
    if abs(ds) > max_accel:
        ds = math.copysign(max_accel, ds)
    ac.speed += ds

    # Altitude dynamics: limited climb/descent rate
    max_climb = 18.0 * dt  # m per step
    dalt = ac.target_alt - ac.alt
    if abs(dalt) > max_climb:
        dalt = math.copysign(max_climb, dalt)
    ac.alt += dalt

    # Kinematic update: heading 0 = north, pi/2 = east
    ac.x += ac.speed * math.sin(ac.heading) * dt
    ac.y += ac.speed * math.cos(ac.heading) * dt


def propagate_one(
    ac: Aircraft,
    dt: float,
    horizon: float,
    override: Optional[dict] = None,
) -> dict:
    x = ac.x
    y = ac.y
    alt = ac.alt
    heading = ac.heading
    speed = ac.speed

    if override:
        target_heading = override.get("target_heading", ac.target_heading)
        target_speed = override.get("target_speed", ac.target_speed)
        target_alt = override.get("target_alt", ac.target_alt)
    else:
        target_heading = ac.target_heading
        target_speed = ac.target_speed
        target_alt = ac.target_alt

    n = int(math.ceil(horizon / dt)) + 1
    t_arr = np.arange(n) * dt
    xs = np.empty(n, dtype=float)
    ys = np.empty(n, dtype=float)
    zs = np.empty(n, dtype=float)

    for k in range(n):
        xs[k] = x
        ys[k] = y
        zs[k] = alt

        # Dynamics for next step
        max_turn = math.radians(3.0) * dt
        dh = normalize_angle(target_heading - heading)
        if abs(dh) > max_turn:
            dh = math.copysign(max_turn, dh)
        heading = normalize_angle(heading + dh)

        max_accel = 0.004 * dt
        ds = target_speed - speed
        if abs(ds) > max_accel:
            ds = math.copysign(max_accel, ds)
        speed += ds

        max_climb = 18.0 * dt
        dalt = target_alt - alt
        if abs(dalt) > max_climb:
            dalt = math.copysign(max_climb, dalt)
        alt += dalt

        x += speed * math.sin(heading) * dt
        y += speed * math.cos(heading) * dt

    return {
        "t": t_arr,
        "x": xs,
        "y": ys,
        "alt": zs,
    }


def predict_world(
    world: Dict[str, Aircraft],
    dt: float,
    horizon: float,
    overrides: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    preds = {}
    for ac_id, ac in world.items():
        ov = None
        if overrides and ac_id in overrides:
            ov = overrides[ac_id]
        preds[ac_id] = propagate_one(ac, dt, horizon, ov)
    return preds


def detect_conflicts(
    preds: Dict[str, dict],
    world: Dict[str, Aircraft],
    h_sep_km: float,
    v_sep_m: float,
    uncertainty_scale: float = 1.0,
) -> List[dict]:
    conflicts = []
    ids = list(preds.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1 = ids[i]
            id2 = ids[j]

            if id1 not in world or id2 not in world:
                continue

            p1 = preds[id1]
            p2 = preds[id2]

            dx = p1["x"] - p2["x"]
            dy = p1["y"] - p2["y"]
            hdist = np.sqrt(dx * dx + dy * dy)
            vdist = np.abs(p1["alt"] - p2["alt"])
            t = p1["t"]

            # Simple growing position uncertainty model
            unc = uncertainty_scale * (
                world[id1].uncertainty_km
                + world[id2].uncertainty_km
                + 0.008 * t
            )

            if uncertainty_scale <= 1e-9:
                risk = np.where(hdist < h_sep_km, 1.0, 0.0)
            else:
                margin = hdist - h_sep_km
                risk = np.clip(1.0 - margin / (unc + 1e-6), 0.0, 1.0)

            vertical_close = vdist < (v_sep_m + 1.0)
            risk = np.where(vertical_close, risk, 0.0)
            risk = np.where((hdist < h_sep_km) & vertical_close, 1.0, risk)

            max_risk = float(np.max(risk)) if risk.size else 0.0

            if max_risk >= CONFLICT_RISK_THRESHOLD:
                idx = int(np.argmax(risk))
                conflicts.append(
                    {
                        "pair": (id1, id2),
                        "time_to_event": float(t[idx]),
                        "risk": max_risk,
                        "hdist_km": float(hdist[idx]),
                        "vsep_m": float(vdist[idx]),
                        "pos1": (
                            float(p1["x"][idx]),
                            float(p1["y"][idx]),
                            float(p1["alt"][idx]),
                        ),
                        "pos2": (
                            float(p2["x"][idx]),
                            float(p2["y"][idx]),
                            float(p2["alt"][idx]),
                        ),
                        "unc_km": float(unc[idx]),
                    }
                )

    conflicts.sort(key=lambda c: c["risk"], reverse=True)
    return conflicts


# =============================================================================
# Resolution planning: candidate generation, simulation, optimization
# =============================================================================

def no_action_candidate(ac: Aircraft) -> dict:
    return {
        "ac_id": ac.id,
        "label": "No action",
        "target_heading": ac.target_heading,
        "target_alt": ac.target_alt,
        "target_speed": ac.target_speed,
        "cost": 0.0,
    }


def maneuver_candidates(ac: Aircraft) -> List[dict]:
    candidates = []

    def add(
        label: str,
        heading_offset_deg: float = 0.0,
        altitude_delta: float = 0.0,
        speed_factor: float = 1.0,
    ):
        target_alt = float(np.clip(ac.nominal_alt + altitude_delta, 8000.0, 12500.0))
        target_speed = float(np.clip(ac.nominal_speed * speed_factor, 0.18, 0.28))
        target_heading = normalize_angle(
            ac.nominal_heading + math.radians(heading_offset_deg)
        )

        cost = (
            abs(heading_offset_deg) / 30.0
            + abs(altitude_delta) / 300.0 * 0.6
            + abs(speed_factor - 1.0) * 25.0
        )

        candidates.append(
            {
                "ac_id": ac.id,
                "label": label,
                "target_heading": target_heading,
                "target_alt": target_alt,
                "target_speed": target_speed,
                "cost": float(cost),
            }
        )

    add("Continue nominal", 0.0, 0.0, 1.0)
    add("Right 15°", 15.0, 0.0, 1.0)
    add("Right 30°", 30.0, 0.0, 1.0)
    add("Left 15°", -15.0, 0.0, 1.0)
    add("Left 30°", -30.0, 0.0, 1.0)
    add("Climb 300 m", 0.0, 300.0, 1.0)
    add("Descend 300 m", 0.0, -300.0, 1.0)
    add("Slow 5%", 0.0, 0.0, 0.95)
    add("Right 15° + Climb", 15.0, 300.0, 1.0)
    add("Left 15° + Descend", -15.0, -300.0, 1.0)

    return candidates


def apply_actions(
    world: Dict[str, Aircraft],
    actions: List[dict],
    sim_time: float,
    logs: Optional[List[dict]] = None,
) -> None:
    for act in actions:
        ac = world.get(act["ac_id"])
        if ac is None:
            continue

        ac.target_heading = act["target_heading"]
        ac.target_alt = act["target_alt"]
        ac.target_speed = act["target_speed"]
        ac.status = "RESOLVING"
        ac.maneuver = act["label"]
        ac.maneuver_start = sim_time

        if logs is not None:
            logs.append(
                {
                    "time": float(sim_time),
                    "msg": (
                        f"{ac.callsign} executing {act['label']} "
                        f"(hdg {rad_to_deg(ac.target_heading):.0f}°, "
                        f"alt {ac.target_alt:.0f} m, "
                        f"spd {ac.target_speed * 1000.0:.0f} m/s)."
                    ),
                }
            )


def plan_resolution(
    world: Dict[str, Aircraft],
    conflicts: List[dict],
    h_sep_km: float,
    v_sep_m: float,
    uncertainty_scale: float,
    lookahead: float = LOOKAHEAD,
    pred_dt: float = PRED_DT,
) -> Tuple[List[dict], dict]:
    if not conflicts:
        return [], {"status": "No conflicts"}

    primary = max(conflicts, key=lambda c: c["risk"])
    id1, id2 = primary["pair"]
    ac1 = world[id1]
    ac2 = world[id2]

    c1_list = [no_action_candidate(ac1)] + maneuver_candidates(ac1)
    c2_list = [no_action_candidate(ac2)] + maneuver_candidates(ac2)

    baseline_pairs = {
        tuple(sorted(c["pair"]))
        for c in conflicts
        if c["risk"] >= CONFLICT_RISK_THRESHOLD
    }
    primary_pair = tuple(sorted((id1, id2)))

    results = []
    best = None

    for c1 in c1_list:
        for c2 in c2_list:
            overrides = {
                id1: {
                    "target_heading": c1["target_heading"],
                    "target_alt": c1["target_alt"],
                    "target_speed": c1["target_speed"],
                },
                id2: {
                    "target_heading": c2["target_heading"],
                    "target_alt": c2["target_alt"],
                    "target_speed": c2["target_speed"],
                },
            }

            preds = predict_world(world, pred_dt, lookahead, overrides)
            new_conflicts = detect_conflicts(
                preds,
                world,
                h_sep_km,
                v_sep_m,
                uncertainty_scale,
            )

            new_pairs = {
                tuple(sorted(c["pair"]))
                for c in new_conflicts
                if c["risk"] >= CONFLICT_RISK_THRESHOLD
            }

            primary_resolved = primary_pair not in new_pairs
            unexpected = new_pairs - baseline_pairs

            max_risk = max((c["risk"] for c in new_conflicts), default=0.0)
            total_risk = sum(c["risk"] for c in new_conflicts)

            cost = c1["cost"] + c2["cost"] + 3.0 * total_risk

            feasible = bool(primary_resolved and not unexpected and max_risk < 0.70)

            rec = {
                "c1": c1,
                "c2": c2,
                "cost": float(cost),
                "max_risk": float(max_risk),
                "total_risk": float(total_risk),
                "feasible": feasible,
                "unexpected": len(unexpected),
                "primary_resolved": primary_resolved,
                "no_action": c1["label"] == "No action" and c2["label"] == "No action",
            }

            results.append(rec)

            if feasible and (best is None or cost < best["cost"]):
                best = rec

    if best is None and results:
        best = min(
            results,
            key=lambda r: (r["unexpected"], r["max_risk"], r["cost"]),
        )

    actions = []
    if best:
        for c in (best["c1"], best["c2"]):
            if c["label"] != "No action":
                actions.append(c)

    alt_rows = []
    sorted_results = sorted(
        results,
        key=lambda r: (not r["feasible"], r["cost"], -r["max_risk"]),
    )

    for r in sorted_results[:12]:
        alt_rows.append(
            {
                "Aircraft 1": r["c1"]["label"],
                "Aircraft 2": r["c2"]["label"],
                "Feasible": r["feasible"],
                "Max risk %": round(r["max_risk"] * 100.0, 1),
                "Cost": round(r["cost"], 2),
                "Primary resolved": r["primary_resolved"],
            }
        )

    report = {
        "primary_pair": f"{id1} ↔ {id2}",
        "primary_risk_pct": round(primary["risk"] * 100.0, 1),
        "evaluated": len(results),
        "feasible_count": sum(1 for r in results if r["feasible"]),
        "best": None
        if best is None
        else {
            "Aircraft 1": best["c1"]["label"],
            "Aircraft 2": best["c2"]["label"],
            "Feasible": best["feasible"],
            "Max risk %": round(best["max_risk"] * 100.0, 1),
            "Cost": round(best["cost"], 2),
            "No action": best["no_action"],
        },
        "alternatives": alt_rows,
    }

    return actions, report


# =============================================================================
# Main simulation update used by both live app and video recorder
# =============================================================================

def update_simulation(
    world: Dict[str, Aircraft],
    histories: Dict[str, dict],
    sim_time: float,
    dt_sim: float,
    auto_resolve: bool,
    next_replan: float,
    h_sep_km: float,
    v_sep_m: float,
    uncertainty_scale: float,
    logs: Optional[List[dict]] = None,
) -> Tuple[float, Dict[str, dict], List[dict], float, Optional[dict]]:
    if dt_sim <= 0.0:
        preds = predict_world(world, PRED_DT, LOOKAHEAD)
        conflicts = detect_conflicts(
            preds,
            world,
            h_sep_km,
            v_sep_m,
            uncertainty_scale,
        )
        return sim_time, preds, conflicts, next_replan, None

    # Substep dynamics for stability
    nsub = max(1, int(math.ceil(dt_sim / 0.75)))
    sub_dt = dt_sim / float(nsub)

    for _ in range(nsub):
        for ac in world.values():
            step_aircraft(ac, sub_dt)

        sim_time += sub_dt

        # Record history approximately once per simulated second
        for ac in world.values():
            h = histories[ac.id]
            if not h["t"] or sim_time - h["t"][-1] >= 1.0:
                h["t"].append(float(sim_time))
                h["x"].append(float(ac.x))
                h["y"].append(float(ac.y))
                h["z"].append(float(ac.alt))

                if len(h["t"]) > 800:
                    h["t"].pop(0)
                    h["x"].pop(0)
                    h["y"].pop(0)
                    h["z"].pop(0)

    preds = predict_world(world, PRED_DT, LOOKAHEAD)
    conflicts = detect_conflicts(
        preds,
        world,
        h_sep_km,
        v_sep_m,
        uncertainty_scale,
    )

    # Recovery logic: return to nominal when safe
    active_ids = set()
    for c in conflicts:
        if c["risk"] >= CONFLICT_RISK_THRESHOLD:
            active_ids.update(c["pair"])

    for ac in world.values():
        if ac.status in ("RESOLVING", "RECOVERING") and ac.id not in active_ids:
            if ac.maneuver_start is not None and sim_time - ac.maneuver_start > 30.0:
                if (
                    abs(normalize_angle(ac.target_heading - ac.nominal_heading)) > 1e-6
                    or abs(ac.target_alt - ac.nominal_alt) > 1.0
                    or abs(ac.target_speed - ac.nominal_speed) > 1e-6
                ):
                    ac.target_heading = ac.nominal_heading
                    ac.target_alt = ac.nominal_alt
                    ac.target_speed = ac.nominal_speed
                    ac.status = "RECOVERING"
                    ac.maneuver = "Return to nominal"

                    if logs is not None:
                        logs.append(
                            {
                                "time": float(sim_time),
                                "msg": f"{ac.callsign} returning to nominal profile.",
                            }
                        )

        if ac.status == "RECOVERING":
            if (
                abs(normalize_angle(ac.heading - ac.nominal_heading)) < math.radians(5.0)
                and abs(ac.alt - ac.nominal_alt) < 60.0
                and abs(ac.speed - ac.nominal_speed) < 0.005
            ):
                ac.status = "NOMINAL"
                ac.maneuver = ""
                ac.maneuver_start = None

                if logs is not None:
                    logs.append(
                        {
                            "time": float(sim_time),
                            "msg": f"{ac.callsign} restored to nominal profile.",
                        }
                    )

    report = None

    if auto_resolve and sim_time >= next_replan:
        if conflicts:
            actions, report = plan_resolution(
                world,
                conflicts,
                h_sep_km,
                v_sep_m,
                uncertainty_scale,
            )

            if actions:
                apply_actions(world, actions, sim_time, logs)
                next_replan = sim_time + REPLAN_COOLDOWN

                if logs is not None and report and report.get("best"):
                    logs.append(
                        {
                            "time": float(sim_time),
                            "msg": (
                                "Optimizer selected "
                                f"{report['best']['Aircraft 1']} / "
                                f"{report['best']['Aircraft 2']} "
                                f"after evaluating {report['evaluated']} candidate pairs."
                            ),
                        }
                    )
            else:
                best_info = report.get("best") if report else None
                if best_info and best_info.get("No action"):
                    next_replan = sim_time + REPLAN_COOLDOWN
                else:
                    next_replan = sim_time + 8.0
        else:
            next_replan = sim_time + 5.0

    return sim_time, preds, conflicts, next_replan, report


# =============================================================================
# Matplotlib frame renderer and downloadable demo video generator
# =============================================================================

def render_matplotlib_frame(
    ax,
    fig,
    world: Dict[str, Aircraft],
    histories: Dict[str, dict],
    preds: Dict[str, dict],
    conflicts: List[dict],
    sim_time: float,
) -> None:
    ax.clear()

    for ac in world.values():
        h = histories[ac.id]

        if h["x"]:
            ax.plot(
                h["x"],
                h["y"],
                h["z"],
                color=ac.color,
                linewidth=1.2,
                alpha=0.85,
            )

        ax.scatter(
            [ac.x],
            [ac.y],
            [ac.alt],
            color=ac.color,
            s=22,
            zorder=5,
        )

        if ac.id in preds:
            p = preds[ac.id]
            ax.plot(
                p["x"],
                p["y"],
                p["alt"],
                color=ac.color,
                linestyle=":",
                linewidth=0.9,
                alpha=0.55,
            )

    for c in conflicts:
        x1, y1, z1 = c["pos1"]
        x2, y2, z2 = c["pos2"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="red",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
        )

        ax.scatter(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="red",
            marker="x",
            s=30,
        )

    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_zlim(8500.0, 11500.0)

    try:
        ax.set_box_aspect((1.0, 1.0, 0.35))
    except Exception:
        pass

    ax.set_title(f"SIMULATION ONLY | t={sim_time:.0f}s", fontsize=8, pad=0)
    ax.set_xlabel("East km", fontsize=6)
    ax.set_ylabel("North km", fontsize=6)
    ax.set_zlabel("Alt m", fontsize=6)
    ax.tick_params(labelsize=6)


def generate_demo_video(duration_sec: int = 20, fps: int = 8) -> Tuple[bytes, str, str]:
    total_frames = max(1, int(duration_sec * fps))
    dt_sim = TOTAL_DEMO_SIM_SECONDS / float(total_frames)

    world = create_demo_world()
    histories = init_histories(world)

    sim_time = 0.0
    next_replan = 0.0
    logs = []

    fig = plt.figure(figsize=(5.2, 3.2), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvas(fig)

    progress_bar = st.progress(0.0, text="Rendering accelerated demo recording...")

    tmp_path = None
    writer = None
    frames = []
    mime = "image/gif"
    ext = "gif"

    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
        mime = "video/mp4"
        ext = "mp4"
    except Exception:
        writer = None
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        tmp_path = None

    for i in range(total_frames):
        sim_time, preds, conflicts, next_replan, _ = update_simulation(
            world=world,
            histories=histories,
            sim_time=sim_time,
            dt_sim=dt_sim,
            auto_resolve=True,
            next_replan=next_replan,
            h_sep_km=H_SEP_KM_DEFAULT,
            v_sep_m=V_SEP_M_DEFAULT,
            uncertainty_scale=1.0,
            logs=logs,
        )

        render_matplotlib_frame(
            ax=ax,
            fig=fig,
            world=world,
            histories=histories,
            preds=preds,
            conflicts=conflicts,
            sim_time=sim_time,
        )

        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

        if writer is not None:
            writer.append_data(img)
        else:
            frames.append(img)

        progress_bar.progress(
            (i + 1) / float(total_frames),
            text=f"Rendering frame {i + 1}/{total_frames}",
        )

    if writer is not None:
        writer.close()
        with open(tmp_path, "rb") as f:
            data = f.read()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    else:
        buf = io.BytesIO()

        try:
            imageio.mimsave(buf, frames, format="GIF", fps=fps)
        except Exception:
            from PIL import Image

            imgs = [Image.fromarray(frame) for frame in frames]
            imgs[0].save(
                buf,
                save_all=True,
                append_images=imgs[1:],
                duration=int(1000.0 / fps),
                loop=0,
            )

        data = buf.getvalue()

    plt.close(fig)
    return data, mime, ext


# =============================================================================
# Streamlit state management and callbacks
# =============================================================================

def add_log(message: str) -> None:
    st.session_state.logs.append(
        {
            "time": float(st.session_state.sim_time),
            "msg": message,
        }
    )

    if len(st.session_state.logs) > 250:
        st.session_state.logs = st.session_state.logs[-250:]


def reset_demo() -> None:
    world = create_demo_world()

    st.session_state.world = world
    st.session_state.histories = init_histories(world)
    st.session_state.sim_time = 0.0
    st.session_state.logs = [
        {
            "time": 0.0,
            "msg": "Demo scenario loaded: 6 aircraft, crossing conflicts expected.",
        }
    ]

    st.session_state.next_replan = 0.0
    st.session_state.running = True
    st.session_state.demo_mode = True
    st.session_state.demo_elapsed_real = 0.0
    st.session_state.demo_complete = False
    st.session_state.candidate_report = None

    st.session_state.detected_count = 0
    st.session_state.resolved_count = 0
    st.session_state.active_conflict_keys = set()

    st.session_state.preds = None
    st.session_state.conflicts = []
    st.session_state.last_real_time = time.time()


def cb_reset() -> None:
    reset_demo()


def cb_toggle() -> None:
    if st.session_state.demo_complete and not st.session_state.running:
        st.session_state.demo_complete = False
        st.session_state.demo_mode = False
        st.session_state.running = True
    else:
        st.session_state.running = not st.session_state.running

    st.session_state.last_real_time = time.time()


def cb_force_replan() -> None:
    st.session_state.next_replan = 0.0
    st.session_state.running = True
    add_log("Manual replan requested.")


def cb_inject_traffic() -> None:
    world = st.session_state.world

    existing = set(world.keys())
    n = 1
    while f"X{n:02d}" in existing:
        n += 1

    ac_id = f"X{n:02d}"
    callsign = f"TRAFFIC{n}"

    angle = random.uniform(0.0, 2.0 * math.pi)
    radius = 75.0

    x = radius * math.sin(angle)
    y = radius * math.cos(angle)

    heading = normalize_angle(angle + math.pi)
    alt = random.choice([9800.0, 10000.0, 10200.0])
    speed = 0.225

    world[ac_id] = Aircraft(
        id=ac_id,
        callsign=callsign,
        x=x,
        y=y,
        alt=alt,
        heading=heading,
        speed=speed,
        target_heading=heading,
        target_speed=speed,
        target_alt=alt,
        nominal_heading=heading,
        nominal_speed=speed,
        nominal_alt=alt,
        origin="USER",
        dest="CENTER",
        color="#e67e22",
        uncertainty_km=0.40,
        status="NOMINAL",
        maneuver="",
        maneuver_start=None,
    )

    st.session_state.histories[ac_id] = {
        "t": [float(st.session_state.sim_time)],
        "x": [float(x)],
        "y": [float(y)],
        "z": [float(alt)],
    }

    add_log(
        f"Injected {callsign} at ({x:.0f}, {y:.0f}) km, alt {alt:.0f} m."
    )


def update_conflict_counts(conflicts: List[dict]) -> None:
    current = set()

    for c in conflicts:
        if c["risk"] >= CONFLICT_RISK_THRESHOLD:
            current.add(tuple(sorted(c["pair"])))

    previous = st.session_state.active_conflict_keys

    for pair in current - previous:
        st.session_state.detected_count += 1
        add_log(f"Predicted conflict detected: {pair[0]} ↔ {pair[1]}")

    for pair in previous - current:
        st.session_state.resolved_count += 1
        add_log(f"Conflict resolved: {pair[0]} ↔ {pair[1]}")

    st.session_state.active_conflict_keys = current


def app_update(real_dt: float) -> None:
    if real_dt <= 0.0:
        return

    dt_sim = real_dt * float(st.session_state.time_scale)

    sim_time, preds, conflicts, next_replan, report = update_simulation(
        world=st.session_state.world,
        histories=st.session_state.histories,
        sim_time=st.session_state.sim_time,
        dt_sim=dt_sim,
        auto_resolve=bool(st.session_state.auto_resolve),
        next_replan=float(st.session_state.next_replan),
        h_sep_km=float(st.session_state.h_sep),
        v_sep_m=float(st.session_state.v_sep),
        uncertainty_scale=float(st.session_state.uncertainty_scale),
        logs=st.session_state.logs,
    )

    st.session_state.sim_time = sim_time
    st.session_state.preds = preds
    st.session_state.conflicts = conflicts
    st.session_state.next_replan = next_replan

    if report is not None:
        st.session_state.candidate_report = report

    if len(st.session_state.logs) > 250:
        st.session_state.logs = st.session_state.logs[-250:]

    update_conflict_counts(conflicts)


# =============================================================================
# Plotly visualization
# =============================================================================

def render_plotly(
    world: Dict[str, Aircraft],
    histories: Dict[str, dict],
    preds: Dict[str, dict],
    conflicts: List[dict],
    show_predictions: bool,
    show_uncertainty: bool,
) -> go.Figure:
    fig = go.Figure()

    for ac_id, ac in world.items():
        h = histories.get(ac_id, {"x": [], "y": [], "z": []})

        fig.add_trace(
            go.Scatter3d(
                x=h["x"],
                y=h["y"],
                z=h["z"],
                mode="lines",
                line=dict(color=ac.color, width=4),
                name=f"{ac.callsign} trail",
                legendgroup=ac_id,
                showlegend=True,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "x: %{x:.1f} km<br>"
                    "y: %{y:.1f} km<br>"
                    "alt: %{z:.0f} m<extra></extra>"
                ),
                text=[ac.callsign] * len(h["x"]),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[ac.x],
                y=[ac.y],
                z=[ac.alt],
                mode="markers+text",
                marker=dict(
                    size=5,
                    color=ac.color,
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                text=[ac.callsign],
                textposition="top center",
                name=f"{ac.callsign} current",
                legendgroup=ac_id,
                showlegend=False,
            )
        )

        if show_predictions and ac_id in preds:
            p = preds[ac_id]

            fig.add_trace(
                go.Scatter3d(
                    x=p["x"],
                    y=p["y"],
                    z=p["alt"],
                    mode="lines",
                    line=dict(color=ac.color, dash="dash", width=2),
                    opacity=0.55,
                    name=f"{ac.callsign} forecast",
                    legendgroup=ac_id,
                    showlegend=False,
                )
            )

            if show_uncertainty:
                step = max(1, len(p["t"]) // 8)
                idx = np.arange(0, len(p["t"]), step)

                unc = float(st.session_state.uncertainty_scale) * (
                    ac.uncertainty_km + 0.008 * p["t"][idx]
                )

                sizes = np.clip(3.0 + 10.0 * unc, 3.0, 18.0)

                fig.add_trace(
                    go.Scatter3d(
                        x=p["x"][idx],
                        y=p["y"][idx],
                        z=p["alt"][idx],
                        mode="markers",
                        marker=dict(
                            size=sizes,
                            color=ac.color,
                            opacity=0.12,
                        ),
                        hoverinfo="skip",
                        legendgroup=ac_id,
                        showlegend=False,
                    )
                )

    for c in conflicts:
        x1, y1, z1 = c["pos1"]
        x2, y2, z2 = c["pos2"]

        fig.add_trace(
            go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                mode="lines",
                line=dict(color="red", width=6, dash="dash"),
                name=f"Conflict {c['pair'][0]}-{c['pair'][1]}",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                mode="markers",
                marker=dict(size=6, color="red", symbol="x"),
                showlegend=False,
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=35, b=0),
        height=560,
        scene=dict(
            xaxis_title="East (km)",
            yaxis_title="North (km)",
            zaxis_title="Altitude (m)",
            xaxis=dict(range=[-90.0, 90.0]),
            yaxis=dict(range=[-90.0, 90.0]),
            zaxis=dict(range=[8500.0, 11500.0]),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.35),
            camera=dict(
                eye=dict(x=1.25, y=-1.25, z=0.55),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="left",
            x=0.0,
        ),
        title=dict(
            text=f"SIMULATION ONLY | sim time {st.session_state.sim_time:.0f} s",
            x=0.5,
            xanchor="center",
        ),
    )

    return fig


# =============================================================================
# Dashboard rendering
# =============================================================================

def render_dashboard() -> None:
    world = st.session_state.world
    conflicts = st.session_state.conflicts or []
    preds = st.session_state.preds or {}

    max_risk = max((c["risk"] for c in conflicts), default=0.0)

    st.caption(
        "Explicit safety constraints: "
        f"horizontal separation ≥ {float(st.session_state.h_sep):.2f} km, "
        f"vertical separation ≥ {float(st.session_state.v_sep):.0f} m, "
        f"conflict-risk threshold ≥ {CONFLICT_RISK_THRESHOLD * 100:.0f}%. "
        "Forecasts include uncertainty inflation."
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("Sim time", f"{st.session_state.sim_time:.0f} s")
    m2.metric(
        "Demo progress",
        f"{min(100.0, st.session_state.demo_elapsed_real / DEMO_REAL_SECONDS * 100.0):.0f}%",
    )
    m3.metric("Aircraft", len(world))
    m4.metric("Active conflicts", len(conflicts))
    m5.metric("Max risk", f"{max_risk * 100.0:.0f}%")
    m6.metric("Resolved", st.session_state.resolved_count)

    st.progress(
        min(1.0, st.session_state.demo_elapsed_real / DEMO_REAL_SECONDS),
        text="80-second demonstration" if st.session_state.demo_mode else "Free-run mode",
    )

    if st.session_state.demo_complete:
        st.info(
            "80-second demonstration completed. Press Start/Pause to continue in free-run mode, "
            "or reset to replay the scripted demonstration."
        )

    fig = render_plotly(
        world=world,
        histories=st.session_state.histories,
        preds=preds,
        conflicts=conflicts,
        show_predictions=bool(st.session_state.show_predictions),
        show_uncertainty=bool(st.session_state.show_uncertainty),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": False,
        },
        key="main_3d_view",
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Predicted trajectory conflicts")

        if conflicts:
            conflict_rows = []

            for c in conflicts:
                conflict_rows.append(
                    {
                        "Aircraft": f"{c['pair'][0]} ↔ {c['pair'][1]}",
                        "ETA (s)": round(c["time_to_event"], 0),
                        "Risk %": round(c["risk"] * 100.0, 1),
                        "Horiz dist (km)": round(c["hdist_km"], 2),
                        "Vert sep (m)": round(c["vsep_m"], 0),
                        "Uncertainty (km)": round(c["unc_km"], 2),
                    }
                )

            st.dataframe(
                pd.DataFrame(conflict_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No predicted conflicts under current separation constraints.")

    with right:
        st.subheader("Fleet status")

        fleet_rows = []

        for ac in world.values():
            fleet_rows.append(
                {
                    "Call": ac.callsign,
                    "Status": ac.status,
                    "Maneuver": ac.maneuver or "-",
                    "Alt (m)": int(ac.alt),
                    "Speed (m/s)": int(round(ac.speed * 1000.0)),
                    "Heading (°)": int(round(rad_to_deg(ac.heading))),
                }
            )

        st.dataframe(
            pd.DataFrame(fleet_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Resolution planner candidate evaluation")

    report = st.session_state.candidate_report

    if report:
        p1, p2, p3 = st.columns(3)

        p1.write(f"**Primary conflict:** {report['primary_pair']}")
        p2.write(f"**Primary risk:** {report['primary_risk_pct']}%")
        p3.write(
            f"**Evaluated:** {report['evaluated']} candidate pairs | "
            f"**Feasible:** {report['feasible_count']}"
        )

        if report.get("best"):
            st.dataframe(
                pd.DataFrame([report["best"]]),
                use_container_width=True,
                hide_index=True,
            )

        if report.get("alternatives"):
            st.dataframe(
                pd.DataFrame(report["alternatives"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No candidate alternatives stored yet.")
    else:
        st.info("No planner report yet. The planner runs automatically when a conflict is detected.")

    st.subheader("Event log")

    log_df = pd.DataFrame(st.session_state.logs[::-1])

    if not log_df.empty:
        log_df["time"] = log_df["time"].round(0)

    st.dataframe(
        log_df,
        use_container_width=True,
        height=220,
        hide_index=True,
    )


# =============================================================================
# Initialize session state
# =============================================================================

st.session_state.setdefault("auto_resolve", True)
st.session_state.setdefault("time_scale", 4.0)
st.session_state.setdefault("h_sep", H_SEP_KM_DEFAULT)
st.session_state.setdefault("v_sep", V_SEP_M_DEFAULT)
st.session_state.setdefault("uncertainty_scale", 1.0)
st.session_state.setdefault("show_predictions", True)
st.session_state.setdefault("show_uncertainty", True)
st.session_state.setdefault("video_seconds", 20)
st.session_state.setdefault("video_fps", 8)

if "world" not in st.session_state:
    reset_demo()


# =============================================================================
# Sidebar controls and video export
# =============================================================================

with st.sidebar:
    st.header("Controls")

    st.checkbox("Auto resolution", key="auto_resolve")
    st.slider("Time scale", min_value=1.0, max_value=8.0, step=0.5, key="time_scale")
    st.slider(
        "Horizontal sep (km)",
        min_value=5.0,
        max_value=12.0,
        step=0.1,
        key="h_sep",
    )
    st.slider(
        "Vertical sep (m)",
        min_value=150.0,
        max_value=600.0,
        step=25.0,
        key="v_sep",
    )
    st.slider(
        "Uncertainty scale",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        key="uncertainty_scale",
    )

    st.checkbox("Forecast paths", key="show_predictions")
    st.checkbox("Uncertainty halo", key="show_uncertainty")

    st.button("Start/Pause", on_click=cb_toggle, use_container_width=True)
    st.button("Reset 80s demo", on_click=cb_reset, use_container_width=True)
    st.button("Force replan now", on_click=cb_force_replan, use_container_width=True)
    st.button("Inject traffic", on_click=cb_inject_traffic, use_container_width=True)

    with st.expander("Demo recorder / video export", expanded=False):
        st.caption(
            "This renders an accelerated replay of the same demo scenario and "
            "produces a downloadable file. MP4 is used when available; otherwise GIF fallback."
        )

        st.slider(
            "Recording length (s)",
            min_value=5,
            max_value=60,
            step=1,
            key="video_seconds",
        )

        st.select_slider(
            "Frames per second",
            options=[5, 8, 10],
            key="video_fps",
        )

        if st.button("Generate demo recording", use_container_width=True):
            with st.spinner("Rendering accelerated demo recording..."):
                data, mime, ext = generate_demo_video(
                    duration_sec=int(st.session_state.video_seconds),
                    fps=int(st.session_state.video_fps),
                )

                st.session_state.video_payload = (data, mime, ext)

        if "video_payload" in st.session_state:
            data, mime, ext = st.session_state.video_payload

            st.download_button(
                "Download demo recording",
                data=data,
                file_name=f"atc_conflict_demo.{ext}",
                mime=mime,
                use_container_width=True,
            )

            if mime == "image/gif":
                st.image(data, caption="Demo recording (GIF fallback)", use_container_width=True)
            else:
                st.video(data)

            st.caption(
                "Recording rendered from the same simulation engine, accelerated to cover "
                "the full 320-second demo scenario."
            )


# =============================================================================
# Header and live fragment
# =============================================================================

st.title(APP_TITLE)
st.warning(DISCLAIMER)


@st.fragment(run_every=0.35)
def live_demo_fragment():
    now = time.time()

    if "last_real_time" not in st.session_state:
        st.session_state.last_real_time = now

    real_dt = min(1.0, max(0.0, now - st.session_state.last_real_time))
    st.session_state.last_real_time = now

    if st.session_state.running and not st.session_state.demo_complete:
        st.session_state.demo_elapsed_real += real_dt

        if (
            st.session_state.demo_mode
            and st.session_state.demo_elapsed_real >= DEMO_REAL_SECONDS
        ):
            st.session_state.running = False
            st.session_state.demo_complete = True
            add_log("80-second demonstration completed. Simulation paused.")
        else:
            app_update(real_dt)
    else:
        if st.session_state.preds is None:
            st.session_state.preds = predict_world(
                st.session_state.world,
                PRED_DT,
                LOOKAHEAD,
            )
            st.session_state.conflicts = detect_conflicts(
                st.session_state.preds,
                st.session_state.world,
                float(st.session_state.h_sep),
                float(st.session_state.v_sep),
                float(st.session_state.uncertainty_scale),
            )

    render_dashboard()


live_demo_fragment()