# Absent mesh slots produce phantom contacts in MuJoCo-Warp 3.11

Minimal collision reproducer for the **MjLab 1.6.0 / MuJoCo 3.11.0 / MuJoCo-Warp 3.11.0** baseline. A mesh whose per-world `geom_dataid` is `-1` should be absent. Stock Warp 3.11 still generates probe contacts for this slot. The guard rejects absent meshes in broadphase, while keeping valid meshes and primitives collidable.

![Measured contacts before and after the fix](media/comparison.gif)

[Side-by-side MP4](media/comparison.mp4) · [Before MP4](media/before.mp4) · [After MP4](media/after.mp4)

The animated preview is embedded as a GIF so GitHub README rendering works without external video hosting; full-resolution MP4 files are included alongside it.

## Reproduce

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this repository root:

```bash
uv sync --locked
uv run --locked scripts/reproduce.py
```

This uses Python 3.11 and a committed lockfile. CPU collision computation is the default; no CUDA compute, Torch, MjLab installation, display server (headless EGL rendering), system FFmpeg, Hugging Face account, motion dataset or robot assets are needed. Initial installation needs network access; subsequent runs reuse the environment and Warp kernel cache. Kernel compilation takes additional time on a cold cache.

Optional CUDA execution, on a machine with a compatible NVIDIA driver:

```bash
uv run --locked scripts/reproduce.py --device cuda:0
```

Outputs:

- `media/{before,after,comparison}.mp4`, `media/comparison.gif`, and a representative PNG.
- `outputs/{before,after}.json`: every measured frame and contact count.
- `outputs/summary.json`: assertions and measured execution time.
- `outputs/{before,after,regression}.log`: backend logs and eight regression cases.

The command exits unsuccessfully unless stock produces absent-mesh contacts, the patch removes them, and the entire large-box motion and valid-slot contact sequence are unchanged. It also verifies a velocity reversal while the small-box ball is still more than 5 cm from visible geometry. It also checks both NXN and SAP_TILE across single-world, broadcast-absent, mixed-world and broadcast-valid layouts, including valid mesh/floor contacts.

## What the animation shows

This is **real free-body dynamics**, using `mujoco_warp.step` with gravity, friction and the constraint/contact solver. Each ball starts at x=0.42 m, radius 0.08 m, horizontal velocity -0.6 m/s and matching rolling angular velocity. After initialization, no positions, velocities, forces or controls are prescribed. Boxes are fixed obstacles. The simulation runs for 2.02 seconds with a 0.005-second physics step; the video plays at half speed.

MuJoCo's native renderer shows two environments in a single scene. The upper blue box is 0.2 m wide, represented by one valid mesh plus an absent padding slot. The lower green box is 0.4 m wide, represented by two adjacent valid meshes. Both worlds share the compiled slot layout. Both boxes remain visible throughout. Only the small world's second mesh data ID is -1.

**Before:** the small-box ball rebounds while still in empty space. **After:** it travels onward until it hits the real small box. The large-box control trajectory is identical before and after. The checker texture makes ball rotation visible. Red coloring and dots indicate actual Warp contacts with the second mesh slot; the cumulative ghost-hit counter keeps the brief false impact visible in the captions. Red dots may be occluded by solid geometry.

Every displayed ball pose comes from recorded simulation qpos, including orientation and vertical displacement. Rendering does not advance physics. `assets/variants.xml` is the dynamic fixture; `assets/render.xml` combines both worlds for display using equivalent solid boxes; `assets/scene.xml` is the original eight-case static regression fixture. Pillow only adds captions and assembles the videos.

The headless renderer defaults to EGL and requires a working host EGL/OpenGL driver (Mesa or NVIDIA), which uv cannot provide. On a desktop supporting GLFW, use `MUJOCO_GL=glfw uv run --locked scripts/reproduce.py`. All fixture geometry is synthetic.

## Patch isolation and scope

`scripts/patch.py` contains the small backend source transformation. The runner verifies the stock 3.11 collision driver's SHA-256 and applies the transformation to an ignored private copy under `.runtime/fixed/`. Stock and patched workers run in separate processes; the installed package and other environments are never modified. All other model state, trajectory and dependencies are identical.

The source guard is needed in **MuJoCo-Warp**, not in MuJoCo's CPU engine. MjLab provides the relevant variant layout; the standalone backend fixture deliberately omits MjLab for fast installation. This repository does not independently test MjLab integration or training behavior. The companion PR materials contain the MjLab integration regression.

This targets a **3.11 backport**, not an assertion that current upstream main still has the same bug. Newer upstream missing-mesh work must be considered when choosing the PR destination. It does not upgrade to 3.15 or change actuator interfaces.

## Observed result

At 0.58 seconds, the stock small-box ball is about 9.9 cm away from the actual box surface but reverses horizontal velocity from -0.600 to +0.106 m/s. After the patch, there is no absent-slot contact; the ball reaches the actual box before rebounding. The entire large-box trajectory matches within 1e-6 m/quaternion component, and all eight static regression cases pass. The command checks these conditions rather than merely drawing labels.

At 2.02 seconds, small-box x is +0.151 m before versus +0.003 m after (both have rebounded by then); large-box x is +0.308 m in both runs. The first full dynamic/video run took 26.65 seconds on this machine, including some kernel compilation, with dependencies already installed. Timing depends on CPU, graphics driver and cache state. This standalone run used CPU physics plus EGL rendering; CUDA physics remains an optional unverified path here.
