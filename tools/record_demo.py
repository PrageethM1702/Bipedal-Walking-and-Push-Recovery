"""
tools/record_demo.py

Renders the showcase to an MP4 -- no window chrome, no cursor, no dropped
frames, unlike a screen capture.

Each frame is rendered offscreen with ``getCameraImage`` and then composited
with a proper HUD drawn in Pillow: phase caption, controller-state badge, live
metrics, a push flash, and 3-D markers (centre of mass, capture point, support
polygon) projected from world space into screen space.

The frames are piped straight into ffmpeg (the copy bundled with
imageio-ffmpeg, so nothing extra needs installing).

Usage
-----
    python tools/record_demo.py                        # ~50 s, 1280x720, 30 fps
    python tools/record_demo.py --seconds 30
    python tools/record_demo.py --width 1080 --height 1080   # square, for feed
    python tools/record_demo.py --out demo.mp4

Run it with an interpreter that has imageio-ffmpeg available, e.g.

    C:\\Users\\panda\\.venv\\Scripts\\python.exe tools/record_demo.py
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sim_manager import SimManager, BACKGROUND
from simulation.robot_interface import RobotInterface
from simulation.push_applicator import PushApplicator
from walking.walk_generator import WalkGenerator
from control.main_controller import MainController
from balance.support_polygon import SupportPolygon
from visualize import PHASES, GAIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FG        = (232, 236, 242)
DIM       = (148, 158, 172)
PANEL     = (18, 20, 25)
ACCENT    = (86, 166, 255)
COM_COL   = (60, 220, 120)
CP_IN     = (86, 166, 255)
CP_OUT    = (255, 74, 74)
POLY_COL  = (255, 205, 60)
PUSH_COL  = (255, 92, 216)
STATE_COL = {
    "STAND":         (120, 170, 235),
    "WALK":          (95, 210, 120),
    "ANKLE_RECOVER": (255, 185, 60),
    "HIP_RECOVER":   (205, 130, 240),
    "STEP_RECOVER":  (255, 90, 80),
    "FALLEN":        (150, 150, 150),
}


def _font(size, bold=False):
    for name in (("seguisb.ttf", "segoeui.ttf") if bold
                 else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_text(draw, text, font, max_w):
    """Trim ``text`` with an ellipsis so it fits inside ``max_w`` pixels."""
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "...", font=font) > max_w:
        text = text[:-1]
    return text.rstrip(" ,.") + "..."


def project(points, view, proj, w, h):
    """World (N,3) -> screen pixels (N,2), plus a validity mask."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    V = np.array(view, dtype=float).reshape(4, 4, order="F")
    P = np.array(proj, dtype=float).reshape(4, 4, order="F")
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    clip = homo @ V.T @ P.T
    ok = clip[:, 3] > 1e-6
    ndc = np.zeros((len(pts), 3))
    ndc[ok] = clip[ok, :3] / clip[ok, 3:4]
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
    ok &= (ndc[:, 2] > -1.0) & (ndc[:, 2] < 1.0)
    return np.stack([sx, sy], axis=1), ok


def make_card(w, h, title, subtitle, bullets=(), footer=""):
    """A full-frame title / summary card."""
    s = max(w, h) / 1280.0
    img = Image.new("RGB", (w, h), tuple(int(c * 255) for c in BACKGROUND))
    d = ImageDraw.Draw(img, "RGBA")

    step = int(64 * s)
    for x in range(0, w, step):
        d.line([x, 0, x, h], fill=(255, 255, 255, 10), width=1)
    for y in range(0, h, step):
        d.line([0, y, w, y], fill=(255, 255, 255, 10), width=1)

    f_title = _font(int(58 * s), bold=True)
    f_sub = _font(int(25 * s))
    f_item = _font(int(24 * s))
    f_foot = _font(int(18 * s))

    y = h * 0.30 if bullets else h * 0.40
    tw = d.textlength(title, font=f_title)
    d.text(((w - tw) / 2, y), title, font=f_title, fill=FG)
    y += 74 * s
    tw = d.textlength(subtitle, font=f_sub)
    d.text(((w - tw) / 2, y), subtitle, font=f_sub, fill=ACCENT)
    y += 62 * s

    if bullets:
        widest = max(d.textlength(t, font=f_item) for t, _ in bullets)
        x0 = (w - (widest + 24 * s)) / 2
        for text, col in bullets:
            d.ellipse([x0, y + 9 * s, x0 + 11 * s, y + 20 * s], fill=col)
            d.text((x0 + 24 * s, y), text, font=f_item, fill=FG)
            y += 42 * s

    if footer:
        tw = d.textlength(footer, font=f_foot)
        d.text(((w - tw) / 2, h - 62 * s), footer, font=f_foot, fill=DIM)
    return np.asarray(img)


class Recorder:
    def __init__(self, out, w, h, fps):
        exe = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = "ffmpeg"
        cmd = [exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
               "-i", "-", "-an",
               "-vcodec", "libx264", "-preset", "slow", "-crf", "19",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.n = 0

    def add(self, img):
        self.proc.stdin.write(img.tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        return self.proc.wait()


def draw_hud(frame, sense, phase, t, total, walking, push, view, proj,
             robot, w, h, minimal=True):
    """Composite the caption (and, unless ``minimal``, the full HUD) onto a
    frame.  Minimal mode is the default: just the phase title and one line of
    description, so the robot itself is the focus."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    s = max(w, h) / 1280.0
    f_title = _font(int(30 * s), bold=True)
    f_body = _font(int(19 * s))
    f_small = _font(int(16 * s))
    f_badge = _font(int(20 * s), bold=True)

    if minimal:
        d.text((34 * s, 30 * s), phase["title"], font=f_title, fill=FG)
        d.text((34 * s, 68 * s),
               fit_text(d, phase["blurb"], f_small, w - 68 * s),
               font=f_small, fill=DIM)
        if push is not None and push[0] < 0.7:
            label = f"push  {push[1]}"
            d.text((34 * s, 104 * s), label, font=f_badge, fill=PUSH_COL)
        return np.asarray(img.convert("RGB"))

    sp = SupportPolygon.from_robot(robot)
    if len(sp.vertices) >= 3:
        v3 = np.array([[x, y, 0.001] for x, y in sp.vertices])
        pix, ok = project(v3, view, proj, w, h)
        if ok.all():
            d.polygon([tuple(q) for q in pix], outline=POLY_COL + (235,),
                      width=max(1, int(2 * s)))

    marks = np.array([[sense["com"][0], sense["com"][1], 0.002],
                      [sense["cp"][0], sense["cp"][1], 0.002]])
    pix, ok = project(marks, view, proj, w, h)
    r = 9 * s
    if ok[0]:
        x, y = pix[0]
        d.line([x - r, y, x + r, y], fill=COM_COL + (255,), width=int(3 * s))
        d.line([x, y - r, x, y + r], fill=COM_COL + (255,), width=int(3 * s))
    if ok[1]:
        x, y = pix[1]
        col = (CP_IN if sense["margin"] >= 0 else CP_OUT) + (255,)
        d.ellipse([x - r, y - r, x + r, y + r], outline=col,
                  width=max(2, int(3 * s)))
        d.line([x - r * 1.6, y, x + r * 1.6, y], fill=col, width=int(2 * s))
        d.line([x, y - r * 1.6, x, y + r * 1.6], fill=col, width=int(2 * s))

    if push is not None:
        age, label, force = push
        a = max(0.0, 1.0 - age / 0.9)
        if a > 0:
            d.rectangle([0, 0, w, h], outline=PUSH_COL + (int(200 * a),),
                        width=int(8 * s))
            tw = d.textlength(f"PUSH  {label}", font=f_badge)
            bx, by = w / 2 - tw / 2 - 18 * s, 92 * s
            d.rounded_rectangle([bx, by, bx + tw + 36 * s, by + 40 * s],
                                radius=8 * s,
                                fill=PUSH_COL + (int(225 * a),))
            d.text((bx + 18 * s, by + 9 * s), f"PUSH  {label}",
                   font=f_badge, fill=(20, 10, 20, int(255 * a)))

    d.rectangle([0, 0, w, 86 * s], fill=PANEL + (215,))
    d.text((30 * s, 16 * s), phase["title"], font=f_title, fill=FG)
    d.text((30 * s, 54 * s),
           fit_text(d, phase["blurb"], f_small, w - 60 * s),
           font=f_small, fill=DIM)

    st = sense["state"].value
    col = STATE_COL.get(st, FG)
    label = f"{st}   |   {'WALKING' if walking else 'STANDING'}"
    tw = d.textlength(label, font=f_badge)
    bx, by = 30 * s, h - 116 * s
    d.rounded_rectangle([bx, by, bx + tw + 34 * s, by + 40 * s],
                        radius=8 * s, fill=(0, 0, 0, 165), outline=col + (255,),
                        width=max(1, int(2 * s)))
    d.ellipse([bx + 14 * s, by + 15 * s, bx + 24 * s, by + 25 * s], fill=col)
    d.text((bx + 32 * s, by + 9 * s), label, font=f_badge, fill=col)

    m = (f"capture-point margin {sense['margin']:+.3f} m     "
         f"CoM speed {sense['speed']:.2f} m/s     "
         f"trunk tilt {sense['tilt']:.2f} rad")
    d.text((30 * s, h - 64 * s), m, font=f_body, fill=DIM)

    lg = [("CoM", COM_COL), ("capture point", CP_IN), ("support polygon", POLY_COL)]
    x = w - 30 * s
    for name, c in reversed(lg):
        tw = d.textlength(name, font=f_small)
        d.text((x - tw, h - 62 * s), name, font=f_small, fill=DIM)
        d.ellipse([x - tw - 22 * s, h - 58 * s, x - tw - 12 * s, h - 48 * s],
                  fill=c)
        x -= tw + 40 * s

    d.rectangle([0, h - 6 * s, w, h], fill=(255, 255, 255, 28))
    d.rectangle([0, h - 6 * s, w * (t / max(total, 1e-6)), h], fill=ACCENT)
    return np.asarray(img.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "showcase.mp4"))
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hud", action="store_true",
                    help="draw the full HUD (badges, metrics, 3-D markers); "
                         "default is plain caption text only")
    ap.add_argument("--no-cards", action="store_true",
                    help="skip the opening/closing title cards")
    args = ap.parse_args()

    W, H, FPS = args.width, args.height, args.fps

    tour = sum(ph["secs"] for ph in PHASES)
    scale = tour / args.seconds
    print(f"[rec] {tour:.0f}s of content -> {args.seconds:.0f}s "
          f"({scale:.2f}x), {W}x{H} @ {FPS}fps")

    sim = SimManager(gui=False)
    rid = sim.init()
    robot = RobotInterface(rid)
    dt = sim.timestep

    walk = WalkGenerator()
    walk.set_walk_parameter(**GAIT)
    walk.generate()
    walk.inverse_kinematics_all()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rec = Recorder(args.out, W, H, FPS)

    proj = p.computeProjectionMatrixFOV(fov=52, aspect=W / H,
                                        nearVal=0.05, farVal=14)
    bg = (np.array(BACKGROUND) * 255).astype(np.uint8)

    ctrl = pusher = None

    def reset():
        nonlocal ctrl, pusher
        p.resetBasePositionAndOrientation(rid, [0, 0, 0.31],
                                          p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
        ctrl = MainController(robot, walk, dt, sit=GAIT["sit"],
                              enable_walk=False)
        robot.reset_motor_angles(ctrl.stand_pose)
        pusher = PushApplicator(robot, dt, duration=0.02)
        for _ in range(int(1.2 / dt)):
            ctrl.step()
            p.stepSimulation()

    reset()

    if not args.no_cards:
        card = make_card(
            W, H,
            "Bipedal Walking & Push Recovery",
            "12-DoF biped  ·  PyBullet  ·  capture-point balance control",
            footer="ankle / hip / stepping strategies  ·  ZMP-style gait generation")
        for _ in range(int(3.0 * FPS)):
            rec.add(card)

    steps_per_frame = max(1, int(round(scale / (FPS * dt))))
    total_frames = int(args.seconds * FPS)
    frame_i = 0
    push = None
    yaw = 48.0

    for ph in PHASES:
        if ph["walk"]:
            ctrl.start_walking()
        else:
            ctrl.stop_walking()
        queue = sorted(ph["pushes"], key=lambda e: e[0])
        n_frames = int(round(ph["secs"] / scale * FPS))
        elapsed = 0.0

        for k in range(n_frames):
            for _ in range(steps_per_frame):
                while queue and queue[0][0] <= elapsed:
                    _, dirn, mag = queue.pop(0)
                    ev = pusher.push(dirn, mag)
                    push = (0.0, ev["label"], ev["force"])
                    print(f"    {ph['name']:9s} PUSH {ev['label']}")
                pusher.update()
                sense = ctrl.step()
                p.stepSimulation()
                elapsed += dt
                if push is not None:
                    push = (push[0] + dt, push[1], push[2])

            if sense["com"][2] < 0.13:
                print(f"    {ph['name']:9s} fell -- resetting")
                reset()
                if ph["walk"]:
                    ctrl.start_walking()

            com = robot.get_com_position()
            yaw += 0.06
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[com[0], com[1], 0.17],
                distance=1.05, yaw=yaw, pitch=-22, roll=0, upAxisIndex=2)
            img = p.getCameraImage(W, H, view, proj, shadow=1,
                                   lightDirection=[1.4, 1.0, 2.4],
                                   renderer=p.ER_TINY_RENDERER)
            rgb = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
            seg = np.reshape(img[4], (H, W))
            rgb[seg < 0] = bg

            walking = ctrl.enable_walk
            rgb = draw_hud(rgb, sense, ph, frame_i / FPS, args.seconds,
                           walking, push if (push and push[0] < 0.9) else None,
                           view, proj, robot, W, H, minimal=not args.hud)
            rec.add(rgb)
            frame_i += 1
            if frame_i % 60 == 0:
                print(f"[rec] {frame_i}/{total_frames} frames "
                      f"({100*frame_i/total_frames:.0f}%)")
        reset()

    if not args.no_cards:
        card = make_card(
            W, H,
            "Results",
            "measured in simulation -- including what does not work",
            bullets=[
                ("walks 1.2 m without falling", STATE_COL["WALK"]),
                ("ankle strategy recovers 25 N pushes, all four directions",
                 STATE_COL["ANKLE_RECOVER"]),
                ("keeps walking through repeated 12 N disturbances",
                 STATE_COL["WALK"]),
                ("stepping recovery blocked by 5 rad/s actuator limit",
                 STATE_COL["STEP_RECOVER"]),
            ],
            footer="analytic leg IK  ·  capture-point stepping  ·  33/33 tests passing")
        for _ in range(int(4.5 * FPS)):
            rec.add(card)

    code = rec.close()
    size = os.path.getsize(args.out) / 1e6 if os.path.exists(args.out) else 0
    print(f"[rec] ffmpeg exit={code}  frames={rec.n}  "
          f"{size:.1f} MB  -> {args.out}")
    sim.disconnect()
    return 0 if code == 0 and size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
