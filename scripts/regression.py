"""Exercise real Warp broadphase with absent/present mesh in heterogeneous worlds."""

import json
import os
from pathlib import Path

import mujoco
import mujoco_warp as mw
import numpy as np
import warp as wp
from mujoco_warp._src.types import BroadphaseType

xml = (Path(__file__).resolve().parents[1] / "assets/scene.xml").read_text()
m = mujoco.MjModel.from_xml_string(xml)
slot = m.geom("slot").id
probe = m.geom("probe").id
floor = m.geom("floor").id
reports = []
with wp.ScopedDevice(os.environ.get("AUDIT_DEVICE", "cpu")):
    for broadphase in [BroadphaseType.NXN, BroadphaseType.SAP_TILE]:
        for layout in [
            "single_missing",
            "broadcast_missing",
            "mixed",
            "broadcast_valid",
        ]:
            broadcast = layout != "mixed"
            nworld = 1 if layout == "single_missing" else 2
            w = mw.put_model(m)
            w.opt.broadphase = broadphase
            ids = np.tile(m.geom_dataid, (1 if broadcast else 2, 1))
            if layout != "broadcast_valid":
                ids[0, slot] = -1
            w.geom_dataid = wp.array(ids, dtype=int)
            d = mw.make_data(m, nworld=nworld, nconmax=64, njmax=64)
            mw.kinematics(w, d)
            mw.collision(w, d)
            wp.synchronize()
            n = int(d.nacon.numpy()[0])
            pairs = d.contact.geom.numpy()[:n]
            worlds = d.contact.worldid.numpy()[:n]
            by = [
                {tuple(sorted(map(int, p))) for p, ww in zip(pairs, worlds) if ww == i}
                for i in range(nworld)
            ]
            sp = tuple(sorted([slot, probe]))
            sf = tuple(sorted([slot, floor]))
            for worldid, contacts in enumerate(by):
                if ids[worldid % len(ids), slot] < 0:
                    assert all(slot not in pair for pair in contacts), (layout, by)
                else:
                    assert sp in contacts and sf in contacts, (layout, by)
            reports.append(
                {
                    "broadphase": int(broadphase),
                    "layout": layout,
                    "contacts": n,
                    "passed": True,
                }
            )
print(json.dumps(reports), flush=True)
