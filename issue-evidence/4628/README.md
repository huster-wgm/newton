# Newton issue 4628: downstream contact evidence

Evidence for https://github.com/newton-physics/newton/issues/4628.
This branch stores demonstration assets only; it is not part of the source-fix PR.

The videos render saved **GPU physics trajectories**, not authored animation.
Newton 1.6.0, MuJoCo / MuJoCo-Warp 3.12.0, Warp 1.17.0.
A 1 kg box (half extents 0.04, 0.025, 0.02 m) is released from z=0.2 m.
The flat heightfield is 0.5 x 0.5 m, 21 x 21 samples (0.025 m spacing).
Gravity only; no policy, PD, control inputs, or terrain packing.
200 physics steps at 0.005 s; 16 identical worlds. Videos show world 0;
all 16 trajectories within each of these runs are identical.

| Case | Actual collision base | Max lateral displacement | Final center z |
| --- | --- | --- | --- |
| Before: stock Newton conversion | 0.0001 m | 0.849369 m | -1.729049 m |
| Restored: original base and derived bounds | 0.1 m | 0.000112 m | 0.019982 m |

The before case moves and rotates after contact, then leaves the finite
heightfield and falls. Its negative final z is **not** evidence of tunnelling
through the center of the terrain. The restored case settles on the surface.
This establishes a downstream consequence of changing the collision volume in
this configuration; it is not a general claim about every timestep, object,
solver version, or G1 rollout determinism.

The restored video is a controlled runtime ablation in Newton 1.6, restoring
only base depth and the associated AABB/radius. It is **not** a rollout of the
newer upstream PR environment. The source PR separately tests conversion
fidelity on current upstream. Rendering does not integrate physics or change
saved poses. The visual model uses the same source surface, box geometry,
and actual base depth; lighting, cameras, grid, cyan surface-extent line, and colors are visualization only.

- `before.mp4`, `restored.mp4`: individual videos.
- `comparison.mp4`, `comparison.gif`: synchronized comparison, before on top.
- `manifest.json`: trajectory hashes, simulation/render timing, and metrics.
- `newton_original`, `newton_restored`: original saved trajectories/reports.
- `minimal_contact_probe.py`: standalone simulation reproduction.
- `render_contact_evidence.py`: renderer (MuJoCo, NumPy, Pillow, ffmpeg required).

Each simulation covers 1 second. Playback is 0.2x, with a one-second final hold.
Both videos use identical fixed orthographic views: a 2.5 m vertical wide view
with 0.5 m grid spacing, and a 0.6 m contact close-up. Camera calibration follows
[MuJoCo's orthographic field-of-view convention](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global).

## Reproduce physics

Use an isolated environment with Newton 1.6.0 / MuJoCo-Warp 3.12.0 / Warp 1.17.0
and a CUDA GPU. Keep the dependency collision capacities unchanged. The probe
checks overflow at every physics step and raises on any overflow.

```sh
python minimal_contact_probe.py --backend newton --terrain heightfield --output newton_original
python minimal_contact_probe.py --backend newton --terrain heightfield --restore-base --output newton_restored
```

Both output directories must not exist before running. For a control that
bypasses Newton entirely, the same script supports `--backend mjw` and
`--base 0.1` / `--base 0.0001`.

## Re-render saved trajectories

Run from this directory, with a fresh output directory and an EGL context:

```sh
MUJOCO_GL=egl python render_contact_evidence.py --input . --output rendered
```
