"""
tools/plot_logs.py

Turns a logged run (see ``tools/data_logger.py``) into a set of PNG figures
written next to the CSVs in ``logs/<run>/plots/``.

Figures produced:
  1. footsteps_planned_vs_actual.png -- top-down map of where the planner
     intended to place each foot vs. where the feet actually touched down,
     with the CoM path and push markers.
  2. footstep_error.png              -- per-step placement error (planned vs
     nearest actual touch-down of the same foot).
  3. com_trajectory.png              -- CoM x/y/z against time.
  4. capture_point.png               -- capture point vs CoM and the support
     margin, the signal the balance controller acts on.
  5. tilt.png                        -- trunk roll/pitch (with push markers).
  6. states.png                      -- controller state timeline.
  7. foot_height.png                 -- foot clearance and contact phases.

Usage:
    python tools/plot_logs.py                # newest run in logs/
    python tools/plot_logs.py logs/run_name  # a specific run
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

STATE_ORDER = ["STAND", "WALK", "ANKLE_RECOVER", "HIP_RECOVER",
               "STEP_RECOVER", "FALLEN"]
STATE_COLOR = {"STAND": "#4C78A8", "WALK": "#54A24B",
               "ANKLE_RECOVER": "#F58518", "HIP_RECOVER": "#B279A2",
               "STEP_RECOVER": "#E45756", "FALLEN": "#7F7F7F"}


def _read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _col(rows, name, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[name]))
        except (KeyError, ValueError):
            out.append(np.nan)
    return np.array(out)


def _push_times(events):
    return [float(e["t"]) for e in events if e.get("kind") == "push"]


def _mark_pushes(ax, pushes, label=True):
    for i, t in enumerate(pushes):
        ax.axvline(t, color="crimson", ls="--", lw=1.2, alpha=0.8,
                   label="push" if (label and i == 0) else None)


def plot_footsteps(steps, ts, pushes, out, events=None):
    """The headline figure: planned vs actual foot placements.

    Style follows the footstep-map convention used in IS-MPC style gait
    reports: large hollow squares for foot placements (so overlapping steps
    stay readable), the step index printed inside each marker, the CoM path
    through them, and an arrow at each push showing the disturbance direction.
    """
    planned = [s for s in steps if s["kind"] == "planned"]
    actual = [s for s in steps if s["kind"] == "actual"]

    fig, ax = plt.subplots(figsize=(11, 8))

    if len(ts):
        ax.plot(_col(ts, "com_x"), _col(ts, "com_y"), "k-",
                lw=2.0, alpha=0.85, label="CoM path", zorder=1)

    if planned:
        ax.scatter([float(s["x"]) for s in planned],
                   [float(s["y"]) for s in planned],
                   marker="s", s=420, facecolors="none",
                   edgecolors="red", linewidths=2.2,
                   label="planned footsteps", zorder=3)
        for i, s in enumerate(planned):
            ax.text(float(s["x"]), float(s["y"]),
                    f"{i}{s['side'][0].upper()}", color="red", fontsize=8,
                    ha="center", va="center", fontweight="bold", zorder=5)
    if actual:
        ax.scatter([float(s["x"]) for s in actual],
                   [float(s["y"]) for s in actual],
                   marker="s", s=420, facecolors="none",
                   edgecolors="darkorange", linewidths=2.2,
                   label="actual footsteps", zorder=4)
        for i, s in enumerate(actual):
            ax.text(float(s["x"]), float(s["y"]) + 0.012,
                    f"{i}{s['side'][0].upper()}", color="darkorange",
                    fontsize=8, ha="center", va="center", fontweight="bold",
                    zorder=5)

    for p in planned:
        same = [a for a in actual if a["side"] == p["side"]
                and float(a["t"]) >= float(p["t"]) - 1e-9]
        if not same:
            continue
        a = min(same, key=lambda s: float(s["t"]))
        ax.plot([float(p["x"]), float(a["x"])], [float(p["y"]), float(a["y"])],
                color="#888888", lw=0.8, alpha=0.6, ls=":", zorder=2)

    if events:
        span = 0.4
        if len(ts):
            span = max(float(np.ptp(_col(ts, "com_x"))),
                       float(np.ptp(_col(ts, "com_y"))), 0.4)
        arrow_len = max(0.06, 0.10 * span)
        first = True
        for e in events:
            if e.get("kind") != "push" or not e.get("fx"):
                continue
            try:
                fx, fy = float(e["fx"]), float(e["fy"])
                px, py = float(e["com_x"]), float(e["com_y"])
            except (KeyError, ValueError):
                continue
            n = np.hypot(fx, fy)
            if n < 1e-8:
                continue
            ax.quiver(px, py, arrow_len * fx / n, arrow_len * fy / n,
                      angles="xy", scale_units="xy", scale=1.0,
                      color="tab:blue", width=0.006, headwidth=3.8,
                      headlength=5.0, zorder=6,
                      label="push impulse" if first else None)
            ax.text(px, py - 0.02, f"{e['detail']}\nt={float(e['t']):.1f}s",
                    color="tab:blue", fontsize=7, ha="center", va="top",
                    zorder=6)
            first = False

    ax.set_xlabel("x  [m]   (forward)")
    ax.set_ylabel("y  [m]   (left)")
    ax.set_title("Footstep map: planned (red) vs actual touch-down (orange)\n"
                 "labels are step index + foot; arrows mark push disturbances")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    handles = [Line2D([], [], color="k", lw=2.0, label="CoM path"),
               Line2D([], [], marker="s", ls="none", markersize=11,
                      markerfacecolor="none", markeredgecolor="red",
                      markeredgewidth=2.0, label="planned footsteps"),
               Line2D([], [], marker="s", ls="none", markersize=11,
                      markerfacecolor="none", markeredgecolor="darkorange",
                      markeredgewidth=2.0, label="actual footsteps")]
    if events and any(e.get("kind") == "push" for e in events):
        handles.append(Line2D([], [], color="tab:blue", lw=3,
                              label="push impulse"))
    ax.legend(handles=handles, loc="best", fontsize=9, labelspacing=1.1,
              borderpad=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_footstep_error(steps, out):
    planned = [s for s in steps if s["kind"] == "planned"]
    actual = [s for s in steps if s["kind"] == "actual"]
    if not planned:
        return False

    labels, errs = [], []
    for i, p in enumerate(planned):
        same = [a for a in actual if a["side"] == p["side"]
                and float(a["t"]) >= float(p["t"]) - 1e-9]
        if not same:
            continue
        a = min(same, key=lambda s: float(s["t"]))
        errs.append(np.hypot(float(a["x"]) - float(p["x"]),
                             float(a["y"]) - float(p["y"])))
        labels.append(f"{i}\n{p['side'][0].upper()}")
    if not errs:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(errs)), errs, color="#4C78A8")
    ax.axhline(float(np.mean(errs)), color="crimson", ls="--",
               label=f"mean {np.mean(errs):.3f} m")
    ax.set_xticks(range(len(errs)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("planned step (index / foot)")
    ax.set_ylabel("placement error [m]")
    ax.set_title("Footstep placement error: planned vs actual")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def plot_com(ts, pushes, out):
    t = _col(ts, "t")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, key, lbl in ((axes[0], "com_x", "x (forward) [m]"),
                         (axes[1], "com_y", "y (lateral) [m]"),
                         (axes[2], "com_z", "z (height) [m]")):
        ax.plot(t, _col(ts, key), color="#4C78A8", lw=1.4)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        _mark_pushes(ax, pushes, label=(ax is axes[0]))
    axes[0].legend(fontsize=8)
    axes[0].set_title("Centre of mass trajectory")
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_capture_point(ts, pushes, out):
    t = _col(ts, "t")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, _col(ts, "com_x"), label="CoM x", color="#4C78A8")
    axes[0].plot(t, _col(ts, "cp_x"), label="capture point x",
                 color="#E45756", ls="--")
    axes[0].plot(t, _col(ts, "com_y"), label="CoM y", color="#54A24B")
    axes[0].plot(t, _col(ts, "cp_y"), label="capture point y",
                 color="#F58518", ls="--")
    axes[0].set_ylabel("position [m]")
    axes[0].set_title("Capture point vs centre of mass")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    _mark_pushes(axes[0], pushes, label=False)

    margin = _col(ts, "cp_margin")
    axes[1].plot(t, margin, color="#B279A2", lw=1.4)
    axes[1].axhline(0.0, color="crimson", lw=1,
                    label="support-polygon edge (step below this)")
    axes[1].fill_between(t, margin, 0, where=(margin < 0),
                         color="crimson", alpha=0.25)
    axes[1].set_ylabel("CP margin [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    _mark_pushes(axes[1], pushes, label=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_tilt(ts, pushes, out):
    t = _col(ts, "t")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, _col(ts, "roll"), label="roll", color="#4C78A8")
    ax.plot(t, _col(ts, "pitch"), label="pitch", color="#F58518")
    ax.plot(t, _col(ts, "speed"), label="CoM speed [m/s]",
            color="#54A24B", alpha=0.7)
    _mark_pushes(ax, pushes)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad] / speed [m/s]")
    ax.set_title("Trunk tilt and CoM speed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_states(ts, pushes, out):
    t = _col(ts, "t")
    states = [r["state"] for r in ts]
    present = [s for s in STATE_ORDER if s in set(states)]
    idx = {s: i for i, s in enumerate(present)}
    y = np.array([idx.get(s, np.nan) for s in states], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.step(t, y, where="post", color="#333333", lw=1.0)
    for s in present:
        m = np.array([st == s for st in states])
        ax.scatter(t[m], y[m], s=8, color=STATE_COLOR.get(s, "#333333"),
                   label=s)
    _mark_pushes(ax, pushes)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_title("Controller state timeline")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_foot_height(ts, pushes, out):
    t = _col(ts, "t")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(t, _col(ts, "lf_z"), label="left foot z", color="#4C78A8")
    axes[0].plot(t, _col(ts, "rf_z"), label="right foot z", color="#F58518")
    axes[0].set_ylabel("foot height [m]")
    axes[0].set_title("Foot clearance and ground contact")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    _mark_pushes(axes[0], pushes, label=False)

    axes[1].fill_between(t, 0, _col(ts, "left_contact"), step="post",
                         alpha=0.6, color="#4C78A8", label="left contact")
    axes[1].fill_between(t, 0, -_col(ts, "right_contact"), step="post",
                         alpha=0.6, color="#F58518", label="right contact")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["right", "", "left"])
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def make_plots(run_dir):
    ts = _read_csv(os.path.join(run_dir, "timeseries.csv"))
    events = _read_csv(os.path.join(run_dir, "events.csv"))
    steps = _read_csv(os.path.join(run_dir, "footsteps.csv"))
    if not ts:
        print(f"[plot_logs] no timeseries in {run_dir}")
        return []

    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    pushes = _push_times(events)
    made = []

    def _out(name):
        made.append(os.path.join(out_dir, name))
        return made[-1]

    plot_footsteps(steps, ts, pushes, _out("footsteps_planned_vs_actual.png"),
                   events=events)
    if not plot_footstep_error(steps, os.path.join(out_dir, "footstep_error.png")):
        pass
    else:
        made.append(os.path.join(out_dir, "footstep_error.png"))
    plot_com(ts, pushes, _out("com_trajectory.png"))
    plot_capture_point(ts, pushes, _out("capture_point.png"))
    plot_tilt(ts, pushes, _out("tilt.png"))
    plot_states(ts, pushes, _out("states.png"))
    plot_foot_height(ts, pushes, _out("foot_height.png"))

    print(f"[plot_logs] wrote {len(made)} figures to {out_dir}")
    return made


def newest_run(log_root=LOG_ROOT):
    if not os.path.isdir(log_root):
        return None
    runs = [os.path.join(log_root, d) for d in os.listdir(log_root)
            if os.path.isdir(os.path.join(log_root, d))]
    return max(runs, key=os.path.getmtime) if runs else None


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    if not target:
        print("No run directory found under logs/. Run a simulation first.")
        sys.exit(1)
    make_plots(target)
