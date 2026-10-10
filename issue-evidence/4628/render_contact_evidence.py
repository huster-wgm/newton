"""Render recorded GPU contact trajectories; never re-simulate their motion.

MUJOCO_GL=egl python -m protomotions.sonic_pp.render_contact_evidence \
    --input outputs/g1/sonic_pp/minimal_contact_v2 --output /tmp/contact-videos
"""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco  # noqa: TID253 - standalone evidence renderer
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # noqa: TID253 - standalone renderer


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def encode(path, frames):
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            "40",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")


def render(case, output, restored):
    report = json.loads((case / "report.json").read_text())
    states = np.load(case / "states.npz")["qpos"]
    assert states.shape == (201, 16, 7)
    assert np.isfinite(states).all()
    base = float(report["runtime_hfield_size"][0][3])
    xml = ET.fromstring((case / "scene.xml").read_text())
    xml.find("asset/hfield").set("size", f".25 .25 .01 {base:.9g}")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="800", offheight="600", orthographic="true", fovy="2.5")
    ET.SubElement(visual, "headlight", ambient=".6 .6 .6", diffuse=".7 .7 .7", specular=".1 .1 .1")
    ET.SubElement(
        xml.find("asset"),
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".13 .18 .25",
        rgb2=".22 .28 .35",
        width="128",
        height="128",
    )
    ET.SubElement(visual, "quality", shadowsize="2048")
    world = xml.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    world.find("geom").set("rgba", ".26 .44 .55 1")
    world.find("body/geom").set("rgba", ".95 .57 .12 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=800)
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    # Identical cameras in both cases. Wide view spans exactly 2.5 m vertically.
    cameras = []
    for lookat, span in (((0, 0, -0.75), 2.5), ((0, 0, 0.08), 0.6)):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.azimuth = 90
        cam.elevation = 0
        cam.orthographic = 1
        cam.distance = 3
        cameras.append((cam, span))

    # Side-view grid, behind the physical scene. Visualization only.
    def view(camera):
        cam, span = camera
        model.vis.global_.fovy = span
        renderer.update_scene(data, camera=cam, scene_option=options)
        scene = renderer.scene
        assert scene.camera[0].orthographic
        for z in np.arange(-2, 0.51, 0.5):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.36, 0.40, 0.46, 1.0]),
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.001, [-1.5, -0.5, z], [1.5, -0.5, z])
            scene.ngeom += 1
        # Mark the finite surface extent even when its true base is sub-pixel.
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.35, 0.85, 1.0, 1.0]),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, [-0.25, 0.251, 0], [0.25, 0.251, 0])
        scene.ngeom += 1
        return Image.fromarray(renderer.render().copy())

    frames = []
    title = "RESTORED | authored base = 0.1 m" if restored else "BEFORE | converted base = 0.0001 m"
    for i, q in enumerate(states[:, 0]):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)  # Kinematic update only; no integration.
        canvas = Image.new("RGB", (1600, 800), "#121923")
        canvas.paste(view(cameras[0]), (0, 140))
        canvas.paste(view(cameras[1]), (800, 140))
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 18), title, font=font(29), fill="#72dcc0" if restored else "#ffad82")
        draw.text(
            (30, 64),
            "Recorded Newton 1.6 / MuJoCo-Warp GPU rollout | gravity only | 0.2x speed",
            font=font(22),
            fill="white",
        )
        draw.text((30, 106), "Wide view: 2.5 m vertical span / 0.5 m grid", font=font(20), fill="#b5c7da")
        draw.text((830, 106), "Contact close-up: same fixed camera in both cases", font=font(20), fill="#b5c7da")
        xy = float(np.linalg.norm(q[:2] - states[0, 0, :2]))
        draw.text(
            (30, 752),
            f"t = {i * 0.005:.3f} s     center z = {q[2]:+.5f} m     lateral displacement = {xy:.6f} m",
            font=font(24),
            fill="white",
        )
        # In the orthographic wide view the ground is 120 px below the top.
        draw.text((18, 248), "z = 0 m", font=font(18), fill="white")
        frames.append(np.asarray(canvas))
        if i in (0, 40, 80, 120, 200):
            canvas.save(output / f"{case.name}_{i:03d}.png")
    renderer.close()
    frames.extend([frames[-1]] * 40)
    name = "restored.mp4" if restored else "before.mp4"
    encode(output / name, frames)
    return {
        "video": name,
        "source": case.name,
        "trajectory_sha256": hashlib.sha256((case / "states.npz").read_bytes()).hexdigest(),
        "actual_base_m": base,
        "max_horizontal_displacement_m": report["max_horizontal_displacement_m"],
        "final_center_z_m": report["final_z_range"][0],
        "versions": report["versions"],
        "frames": len(frames),
        "video_fps": 40,
        "simulation_dt": 0.005,
        "playback_speed": 0.2,
        "world_shown": 0,
        "simulated_worlds": 16,
        "rendering_only": True,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = [
        render(args.input / name, args.output, restored)
        for name, restored in [("newton_original", False), ("newton_restored", True)]
    ]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(args.output / "before.mp4"),
            "-i",
            str(args.output / "restored.mp4"),
            "-filter_complex",
            "[0:v]scale=800:400[a];[1:v]scale=800:400[b];[a][b]vstack[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output / "comparison.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(args.output / "comparison.mp4"),
            "-filter_complex",
            "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(args.output / "comparison.gif"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
