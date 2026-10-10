"""Standalone rigid-body contact probe. No SONIC, GRAIL, PD or terrain packing.

Native MuJoCo-Warp uses only public put_model/put_data/step APIs. Dependencies
are not patched; the upstream heightfield candidate capacity remains unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco  # noqa: TID253 - standalone evidence script
import mujoco_warp as mjw  # noqa: TID253 - standalone evidence script
import numpy as np
import warp as wp


def scene(terrain, shape, base):
    asset = (
        '<asset><hfield name="ground" nrow="21" ncol="21" size=".25 .25 .01 ' + str(base) + '"/></asset>'
        if terrain == "heightfield"
        else ""
    )
    floor = {
        "none": "",
        "plane": '<geom name="ground" type="plane" size=".25 .25 .1"/>',
        "heightfield": '<geom name="ground" type="hfield" hfield="ground"/>',
    }[terrain]
    geom = 'type="box" size=".04 .025 .02"' if shape == "box" else 'type="sphere" size=".04"'
    collider = f'<geom name="object" {geom} mass="1"/>'
    if shape == "compound":
        collider = "".join(
            f'<geom name="part{i}" type="box" size=".02 .025 .02" pos="{x} 0 0" mass=".5"/>'
            for i, x in enumerate((-0.02, 0.02))
        )
    return f'<mujoco><option timestep=".005" gravity="0 0 -9.81" integrator="Euler"/><default><geom friction=".5 .005 .0001"/></default>{asset}<worldbody>{floor}<body name="object" pos="0 0 .2"><freejoint/>{collider}</body></worldbody></mujoco>'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--backend", choices=["mjw", "newton", "cpu"], required=True)
    p.add_argument("--terrain", choices=["none", "plane", "heightfield"], required=True)
    p.add_argument("--shape", choices=["box", "sphere", "compound"], default="box")
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--base", type=float, default=0.1)
    p.add_argument("--model", type=Path, help="Load an exact exported .mjb model for a direct backend comparison")
    p.add_argument("--restore-base", action="store_true", help="Newton-only: restore authored base, AABB and radius")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    xml = scene(a.terrain, a.shape, a.base)
    (a.output / "scene.xml").write_text(xml)
    native = mujoco.MjModel.from_binary_path(str(a.model)) if a.model else mujoco.MjModel.from_xml_string(xml)
    initial = mujoco.MjData(native)
    mujoco.mj_forward(native, initial)
    meta = {
        "backend": a.backend,
        "terrain": a.terrain,
        "shape": a.shape,
        "worlds": a.worlds,
        "versions": {"mujoco": mujoco.__version__, "warp": wp.__version__},
        "dt": 0.005,
        "source_hfield_size": native.hfield_size.tolist(),
        "source_geom_friction": native.geom_friction.tolist(),
        "no_dependency_patches": True,
        "no_network_or_control": True,
    }
    meta.update(restore_base=a.restore_base, input_model=str(a.model) if a.model else None)
    states = []
    contact_counts = []
    first_contacts = None
    begin = time.perf_counter()
    if a.backend == "cpu":
        ds = [mujoco.MjData(native) for _ in range(a.worlds)]
        for tick in range(a.steps + 1):
            states.append(np.stack([d.qpos.copy() for d in ds]))
            if tick == a.steps:
                break
            for d in ds:
                mujoco.mj_step(native, d)
            contact_counts.append(sum(d.ncon for d in ds))
    else:
        with wp.ScopedDevice("cuda:0"):
            if a.backend == "mjw":
                m = mjw.put_model(native)
                d = mjw.put_data(native, initial, nworld=a.worlds, nconmax=64, njmax=256)

                def read():
                    return d.qpos.numpy()

                def step():
                    mjw.step(m, d)
            else:
                import newton  # noqa: PLC0415 - only the Newton backend needs this dependency

                meta["versions"]["newton"] = newton.__version__
                builder = newton.ModelBuilder()
                newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
                template = newton.ModelBuilder()
                newton.solvers.SolverMuJoCo.register_custom_attributes(template)
                template.default_shape_cfg.gap = 0.0
                template.add_mjcf(str(a.output / "scene.xml"), ctrl_direct=True, collapse_fixed_joints=False)
                for _ in range(a.worlds):
                    builder.add_world(template)
                model = builder.finalize(device="cuda:0")
                solver = newton.solvers.SolverMuJoCo(
                    model,
                    separate_worlds=True,
                    use_mujoco_cpu=False,
                    use_mujoco_contacts=True,
                    update_data_interval=1,
                    nconmax=64,
                    njmax=256,
                    save_to_mjcf=str(a.output / "newton_generated.xml"),
                )
                state = model.state()
                next_state = model.state()
                control = model.control()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                m = solver.mjw_model
                d = solver.mjw_data
                mujoco.mj_saveModel(solver.mj_model, str(a.output / "newton_export.mjb"), None)
                if a.restore_base:
                    sizes = m.hfield_size.numpy().copy()
                    sizes[:, 3] = a.base
                    m.hfield_size.assign(sizes)
                    bounds = m.geom_aabb.numpy().copy()
                    radius = m.geom_rbound.numpy().copy()
                    g = int(np.flatnonzero(solver.mj_model.geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD))[0])
                    sx, sy, top, base = sizes[0]
                    bounds[..., g, 0, :] = [0, 0, (top - base) / 2]
                    bounds[..., g, 1, :] = [sx, sy, (top + base) / 2]
                    radius[..., g] = np.linalg.norm([sx, sy, max(top, base)])
                    m.geom_aabb.assign(bounds)
                    m.geom_rbound.assign(radius)

                def read():
                    q = state.joint_q.numpy().reshape(a.worlds, 7)
                    return q[:, [0, 1, 2, 6, 3, 4, 5]]

                def step():
                    nonlocal state, next_state
                    state.clear_forces()
                    solver.step(state, next_state, control, None, 0.005)
                    state, next_state = next_state, state

            meta["runtime_hfield_size"] = m.hfield_size.numpy().tolist()
            meta["runtime_geom_friction"] = m.geom_friction.numpy().tolist()
            for tick in range(a.steps + 1):
                q = read()
                states.append(q.copy())
                if not np.isfinite(q).all():
                    raise RuntimeError("Nonfinite qpos")
                if tick == a.steps:
                    break
                step()
                overflow = d.overflow.numpy()
                if overflow.any():
                    raise RuntimeError(f"Physics overflow at {tick}: {overflow.tolist()}")
                count = int(d.nacon.numpy()[0])
                contact_counts.append(count)
                if count and first_contacts is None:
                    first_contacts = {
                        "tick": tick,
                        "count": count,
                        "world": d.contact.worldid.numpy()[:count].tolist(),
                        "dist": d.contact.dist.numpy()[:count].tolist(),
                        "pos": d.contact.pos.numpy()[:count].tolist(),
                        "frame": d.contact.frame.numpy()[:count].tolist(),
                    }
    q = np.asarray(states)
    pos = np.linalg.norm(q[:, :, :3] - q[:, :1, :3], axis=-1)
    quat = q[:, :, 3:].astype(np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    chord = np.minimum(np.linalg.norm(quat - quat[:, :1], axis=-1), np.linalg.norm(quat + quat[:, :1], axis=-1))
    angle = 4 * np.arcsin(np.clip(chord / 2, 0, 1))
    meta.update(
        elapsed_s=time.perf_counter() - begin,
        steps=a.steps,
        first_position_divergence_step=next((i for i, v in enumerate(pos.max(1)) if v > 1e-6), None),
        max_duplicate_position_m=float(pos.max()),
        max_duplicate_angle_rad=float(angle.max()),
        max_duplicate_raw_qpos=float(np.max(np.abs(q - q[:, :1]))),
        final_z_range=[float(q[-1, :, 2].min()), float(q[-1, :, 2].max())],
        max_horizontal_displacement_m=float(np.linalg.norm(q[:, :, :2] - q[:1, :, :2], axis=-1).max()),
        contact_count_range=[min(contact_counts, default=0), max(contact_counts, default=0)],
    )
    np.savez_compressed(a.output / "states.npz", qpos=q, fps=200)
    (a.output / "first_contacts.json").write_text(json.dumps(first_contacts, indent=2) + "\n")
    (a.output / "report.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
