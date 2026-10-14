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

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]


def worker(mode, device):
    if mode == "after":
        sys.path.insert(0, str(ROOT / ".runtime/fixed"))
    import mujoco
    import mujoco_warp as mw
    import numpy as np
    import warp as wp
    from mujoco_warp._src.types import BroadphaseType

    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/variants.xml"))
    slot, probe = m.geom("slot").id, m.geom("probe").id
    frames = []
    with wp.ScopedDevice(device):
        w = mw.put_model(m)
        w.opt.broadphase = BroadphaseType.NXN
        ids = np.tile(m.geom_dataid, (2, 1))
        ids[0, slot] = -1
        w.geom_dataid = wp.array(ids, dtype=int)
        d = mw.make_data(m, nworld=2, nconmax=64, njmax=64)
        for x in np.linspace(-0.36, 0.30, 81):
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
            positions = d.contact.pos.numpy()[:n]
            base = m.geom("base").id
            base_hits = [
                sum(
                    set(map(int, pair)) == {base, probe}
                    for pair, world in zip(pairs, worlds)
                    if world == i
                )
                for i in range(2)
            ]
            points = [
                [
                    pos.tolist()
                    for pair, world, pos in zip(pairs, worlds, positions)
                    if world == i and set(map(int, pair)) == {slot, probe}
                ]
                for i in range(2)
            ]
            frames.append(
                {
                    "x": float(x),
                    "probe_mesh_contacts": hit,
                    "base_contacts": base_hits,
                    "contact_points": points,
                }
            )
    target = ROOT / "outputs" / f"{mode}.json"
    target.write_text(
        json.dumps({"mode": mode, "device": device, "frames": frames}, indent=2)
    )


def render(traces):
    """MuJoCo renders one scene with both worlds; Pillow only adds captions."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/render.xml"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.08, 0, 0.04]
    camera.distance = 1.95
    camera.azimuth = 90
    camera.elevation = -62
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    images = []
    writers = {
        mode: imageio.get_writer(
            str(ROOT / "media" / f"{mode}.mp4"),
            fps=20,
            codec="libx264",
            macro_block_size=1,
        )
        for mode in traces
    }
    combined = imageio.get_writer(
        str(ROOT / "media/comparison.mp4"), fps=20, codec="libx264", macro_block_size=1
    )
    with mujoco.Renderer(model, height=600, width=720) as renderer:
        try:
            for index in range(len(traces["before"]["frames"])):
                panels = []
                for mode, trace in traces.items():
                    f = trace["frames"][index]
                    for world in range(2):
                        data.qpos[
                            model.jnt_qposadr[model.joint(f"probe{world}").id]
                        ] = f["x"]
                    for world in range(2):
                        model.geom(f"probe_geom{world}").rgba[:] = (
                            [1.0, 0.12, 0.12, 1.0]
                            if f["probe_mesh_contacts"][world]
                            else [0.98, 0.65, 0.12, 1.0]
                        )
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    # Contact markers use actual Warp contact positions, not CPU contacts.
                    for world in range(2):
                        for point in f["contact_points"][world]:
                            pos = np.array(point) + [
                                0,
                                0.36 if world == 0 else -0.36,
                                0,
                            ]
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(
                                geom,
                                mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.012] * 3),
                                pos,
                                np.eye(3).ravel(),
                                np.array([1.0, 0.12, 0.12, 1.0], dtype=np.float32),
                            )
                            renderer.scene.ngeom += 1
                    im = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, 720, 62), fill="#101827")
                    draw.text(
                        (20, 10),
                        f"{mode.upper()} | MuJoCo-Warp 3.11 "
                        + ("stock" if mode == "before" else "+ collision guard"),
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (20, 36),
                        "One scene / two worlds / same prescribed probe sweep",
                        font=small,
                        fill="#b5c3d9",
                    )
                    draw.rectangle((0, 514, 720, 600), fill="#101827")
                    draw.text(
                        (20, 524),
                        f"Small box: padded-slot contacts = {f['probe_mesh_contacts'][0]}",
                        font=font,
                        fill="#ff677e" if f["probe_mesh_contacts"][0] else "#6ce4b5",
                    )
                    draw.text(
                        (20, 551),
                        f"Large box: valid-slot contacts = {f['probe_mesh_contacts'][1]}",
                        font=font,
                        fill="#b5c3d9",
                    )
                    draw.text(
                        (20, 580),
                        f"Red probe/dots: measured slot contact | frame {index + 1}/81",
                        font=small,
                        fill="#b5c3d9",
                    )
                    writers[mode].append_data(np.array(im))
                    panels.append(im)
                both = Image.new("RGB", (1440, 600))
                both.paste(panels[0], (0, 0))
                both.paste(panels[1], (720, 0))
                combined.append_data(np.array(both))
                images.append(both)
            # Select a phantom-contact frame with the probe clear of the small box.
            idx = next(
                i
                for i, f in enumerate(traces["before"]["frames"])
                if f["probe_mesh_contacts"][0] and f["x"] > 0.025
            )
            images[idx].save(ROOT / "media/contact-frame.png")
            previews = [im.resize((960, 400)) for im in images]
            previews[0].save(
                ROOT / "media/comparison.gif",
                save_all=True,
                append_images=previews[1:],
                duration=50,
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
    assert [f["base_contacts"] for f in traces["before"]["frames"]] == [
        f["base_contacts"] for f in traces["after"]["frames"]
    ]
    assert all(
        any(f["base_contacts"][w] for f in traces["after"]["frames"]) for w in range(2)
    )
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
