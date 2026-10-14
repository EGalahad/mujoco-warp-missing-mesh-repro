"""Run independent stock/fixed processes, assert contacts, render measured traces."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(mode, device):
    if mode == "after":
        sys.path.insert(0, str(ROOT / ".runtime/fixed"))
    import mujoco
    import mujoco_warp as mw
    import numpy as np
    import warp as wp
    from mujoco_warp._src.types import BroadphaseType

    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene.xml"))
    slot, probe = m.geom("slot").id, m.geom("probe").id
    frames = []
    with wp.ScopedDevice(device):
        w = mw.put_model(m)
        w.opt.broadphase = BroadphaseType.NXN
        ids = np.tile(m.geom_dataid, (2, 1))
        ids[0, slot] = -1
        w.geom_dataid = wp.array(ids, dtype=int)
        d = mw.make_data(m, nworld=2, nconmax=64, njmax=64)
        for x in np.linspace(-0.30, 0.30, 61):
            q = np.tile(m.qpos0, (2, 1))
            q[:, m.jnt_qposadr[m.body("robot").jntadr[0]]] = x
            d.qpos.assign(q)
            mw.kinematics(w, d)
            mw.collision(w, d)
            wp.synchronize()
            n = int(d.nacon.numpy()[0])
            pairs = d.contact.geom.numpy()[:n]
            worlds = d.contact.worldid.numpy()[:n]
            hit = [
                sum(
                    set(map(int, pair)) == {slot, probe}
                    for pair, world in zip(pairs, worlds)
                    if world == i
                )
                for i in range(2)
            ]
            frames.append({"x": float(x), "probe_mesh_contacts": hit})
    target = ROOT / "outputs" / f"{mode}.json"
    target.write_text(
        json.dumps({"mode": mode, "device": device, "frames": frames}, indent=2)
    )


def render(traces):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    images = []
    writers = {
        mode: imageio.get_writer(
            str(ROOT / "media" / f"{mode}.mp4"),
            fps=15,
            codec="libx264",
            macro_block_size=1,
        )
        for mode in traces
    }
    combined = imageio.get_writer(
        str(ROOT / "media/comparison.mp4"), fps=15, codec="libx264", macro_block_size=1
    )
    try:
        for index in range(61):
            panels = []
            for mode, trace in traces.items():
                im = Image.new("RGB", (560, 440), "#101827")
                draw = ImageDraw.Draw(im)
                draw.text(
                    (20, 14),
                    f"{mode.upper()}: Warp 3.11 "
                    + ("stock" if mode == "before" else "+ guard"),
                    font=font,
                    fill="white",
                )
                f = trace["frames"][index]
                for world in range(2):
                    top = 60 + world * 170
                    draw.text(
                        (20, top),
                        "ABSENT mesh (dataid = -1)"
                        if world == 0
                        else "VALID mesh (control)",
                        font=font,
                        fill="#b5c3d9",
                    )
                    cx, cy, scale = 280, top + 85, 550
                    if world:
                        draw.rectangle(
                            (cx - 55, cy - 55, cx + 55, cy + 55), fill="#426a8f"
                        )
                    else:
                        draw.line((cx - 6, cy, cx + 6, cy), fill="#526176", width=1)
                        draw.line((cx, cy - 6, cx, cy + 6), fill="#526176", width=1)
                    px, radius = cx + f["x"] * scale, 0.08 * scale
                    hit = f["probe_mesh_contacts"][world]
                    draw.ellipse(
                        (px - radius, cy - radius, px + radius, cy + radius),
                        fill="#efab43",
                        outline="#ffd78b",
                        width=2,
                    )
                    if hit:
                        draw.line((px, cy, cx, cy), fill="#ff5a71", width=4)
                    draw.text(
                        (20, top + 140),
                        f"Measured probe/mesh contacts: {hit}",
                        font=small,
                        fill="#ff5a71" if hit else "#6ce4b5",
                    )
                draw.text(
                    (20, 412),
                    f"Top view | prescribed motion | frame {index + 1}/61",
                    font=small,
                    fill="#b5c3d9",
                )
                writers[mode].append_data(np.array(im))
                panels.append(im)
            both = Image.new("RGB", (1120, 440))
            both.paste(panels[0], (0, 0))
            both.paste(panels[1], (560, 0))
            combined.append_data(np.array(both))
            images.append(both)
        images[30].save(ROOT / "media/contact-frame.png")
        images[0].save(
            ROOT / "media/comparison.gif",
            save_all=True,
            append_images=images[1:],
            duration=67,
            loop=0,
        )
    finally:
        combined.close()
        for writer in writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker", choices=["before", "after"])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.device)
        return
    start = time.monotonic()
    from patch import patched

    source = Path(importlib.util.find_spec("mujoco_warp").origin).parent
    driver = source / "_src/collision_driver.py"
    digest = hashlib.sha256(driver.read_bytes()).hexdigest()
    assert (
        digest == "760f21aaeeb156929965ffa5645399dde5f8bf79e3f6a502d5278b9ab949c6ba"
    ), digest
    fixed = ROOT / ".runtime/fixed/mujoco_warp"
    shutil.copytree(
        source, fixed, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
    )
    (fixed / "_src/collision_driver.py").write_text(patched(driver.read_text()))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "media").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["AUDIT_DEVICE"] = args.device
    for mode in ["before", "after"]:
        with (ROOT / "outputs" / f"{mode}.log").open("w") as log:
            subprocess.run(
                [sys.executable, __file__, "--worker", mode, "--device", args.device],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    traces = {
        mode: json.loads((ROOT / "outputs" / f"{mode}.json").read_text())
        for mode in ["before", "after"]
    }
    counts = {
        mode: [
            sum(f["probe_mesh_contacts"][world] > 0 for f in trace["frames"])
            for world in range(2)
        ]
        for mode, trace in traces.items()
    }
    assert counts["before"][0] > 0, counts
    assert counts["after"][0] == 0, counts
    assert counts["after"][1] > 0, counts
    assert [f["probe_mesh_contacts"][1] for f in traces["before"]["frames"]] == [
        f["probe_mesh_contacts"][1] for f in traces["after"]["frames"]
    ]
    env["PYTHONPATH"] = str(ROOT / ".runtime/fixed")
    with (ROOT / "outputs/regression.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/regression.py")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    render(traces)
    report = {
        "frames_with_contacts_absent_valid": counts,
        "regression_cases": 8,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "device": args.device,
        "stock_source_sha256": digest,
    }
    (ROOT / "outputs/summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
