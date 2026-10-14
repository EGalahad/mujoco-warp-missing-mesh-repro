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

The command exits unsuccessfully unless stock produces absent-mesh contacts, the patch removes them, and the entire valid-mesh contact sequence is unchanged. It also checks both NXN and SAP_TILE across single-world, broadcast-absent, mixed-world and broadcast-valid layouts, including valid mesh/floor contacts.

## What the animation shows

MuJoCo's native offscreen renderer draws **one scene containing two environments** in each before/after video, including shadows, lighting and perspective. Both environments always contain a visible box. The upper blue box is 0.2 m wide; the lower green box is 0.4 m wide. Both probes follow the same prescribed 81-position sweep. The combined video places the stock and patched scenes side by side.

The small box uses one valid cube mesh plus one absent padding slot. The large box uses two adjacent valid cube meshes forming a rectangular box. Both share the same compiled two-slot layout; only the small world's second mesh ID is -1. This reproduces variable collision-part counts without making either object disappear. The renderer uses equivalent solid box geometry for the visible union of valid meshes.

Each frame calls real MuJoCo-Warp `kinematics` and `collision`. A red probe indicates a measured contact with the second slot; red dots mark its actual Warp contact positions (sometimes occluded by solid geometry). Thus a red probe beside the small box demonstrates a phantom padding-slot contact. Large-box slot contacts remain after the fix. Contacts with the valid first mesh are also checked and remain identical before/after.

This is prescribed-motion collision inspection, **not a free-dynamics rollout**: probes may pass through boxes because no dynamics step is requested. Only caption text and video composition use Pillow; all scene geometry is rendered by MuJoCo. `assets/variants.xml` is the physics fixture, `assets/render.xml` the two-environment display scene, and `assets/scene.xml` the original eight-case regression fixture. All geometry is synthetic.

The default headless renderer uses EGL and needs a working system EGL/OpenGL driver (Mesa or NVIDIA); these system drivers cannot be installed through uv. On a desktop supporting GLFW, use `MUJOCO_GL=glfw uv run --locked scripts/reproduce.py`. If EGL initialization fails, configure the host graphics driver rather than changing the collision dependencies.

## Patch isolation and scope

`scripts/patch.py` contains the small backend source transformation. The runner verifies the stock 3.11 collision driver's SHA-256 and applies the transformation to an ignored private copy under `.runtime/fixed/`. Stock and patched workers run in separate processes; the installed package and other environments are never modified. All other model state, trajectory and dependencies are identical.

The source guard is needed in **MuJoCo-Warp**, not in MuJoCo's CPU engine. MjLab provides the relevant variant layout; the standalone backend fixture deliberately omits MjLab for fast installation. This repository does not independently test MjLab integration or training behavior. The companion PR materials contain the MjLab integration regression.

This targets a **3.11 backport**, not an assertion that current upstream main still has the same bug. Newer upstream missing-mesh work must be considered when choosing the PR destination. It does not upgrade to 3.15 or change actuator interfaces.

## Observed result

The full 81-frame native-render command took 14.74 seconds on the development machine with warm Warp kernel caches, excluding installation. Stock produced absent-slot contacts in 19 frames; patched produced zero. The large box's second mesh produced contacts in 44 frames in both runs, with identical per-frame counts. Valid first-mesh contacts also matched in both worlds. All eight backend regression cases passed. Cold compilation, graphics drivers and download time vary; this is not a cold-start guarantee. Collision execution was verified on CPU; native rendering used EGL. CUDA collision execution has not been validated in this standalone environment.
