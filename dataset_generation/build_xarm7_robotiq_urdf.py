#!/usr/bin/env python
"""Graft the Robotiq 2F-85 gripper onto the bare xArm7 arm.

ManiSkill ships an ``xarm7`` arm (no gripper) and, separately, an ``xarm6_robotiq``
agent (xArm6 + Robotiq 2F-85). The hardware target here is an xArm7 wearing the same
Robotiq 2F-85 gripper, but no combined URDF exists. Both UFACTORY arms expose the
identical ISO tool flange, and in both stock URDFs the gripper bolts to the flange
link with an *identity* ``gripper_fix`` joint — so the 2F-85 subtree transplants
cleanly onto xArm7's ``link_eef``.

This script emits a self-contained ``xarm7_robotiq.urdf`` (+ matching ``.srdf``):

* arm links/joints come from ``xarm7.urdf`` (PACKAGE_ASSET_DIR, ships with the wheel);
* the gripper subtree (``robotiq_arg2f_base_link`` and below, plus the ``eef`` TCP
  link 0.15 m past the flange) comes from ``xarm6_robotiq.urdf`` (ASSET_DIR — run
  ``python -m mani_skill.utils.download_asset xarm6`` first);
* ``gripper_fix`` is reparented ``link6`` → ``link_eef`` (still identity);
* mesh ``filename`` attributes keep their ORIGINAL relative prefixes — ``meshes/…``
  (arm) and ``meshes_robotiq/…`` (gripper), which don't collide — and the generator
  drops two symlinks beside the URDF pointing at the two source mesh trees. This is
  required because mplib resolves mesh paths relative to the URDF directory and (unlike
  SAPIEN) ignores absolute paths, concatenating them onto the URDF dir instead.

Machine-specific (the symlinks target the local asset install); regenerate after
re-downloading assets or moving the repo:

    python dataset_generation/build_xarm7_robotiq_urdf.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Gripper subtree root + the TCP link; everything reachable from the root through
# the joints listed below is transplanted. The camera_link on xarm6 is intentionally
# left behind (xArm6-specific).
GRIPPER_ROOT_LINK = "robotiq_arg2f_base_link"
GRIPPER_TCP_LINK = "eef"
GRIPPER_LINKS = [
    "robotiq_arg2f_base_link",
    "eef",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
    "right_inner_knuckle",
]
GRIPPER_JOINTS = [
    "gripper_fix",
    "eef_joint",
    "left_outer_knuckle_joint",
    "left_outer_finger_joint",
    "left_inner_knuckle_joint",
    "left_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_outer_knuckle_joint",
    "right_outer_finger_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_pad_joint",
]
ARM_FLANGE_LINK = "link_eef"  # xArm7 flange the gripper bolts to.


def _mesh_prefix(links) -> str:
    """Single top-level mesh dir shared by every <mesh> under the given links.

    Scans only the supplied link elements (not the whole URDF) so an arm's own mesh
    dirs don't leak in when we only transplant a gripper subtree.
    """
    prefixes = set()
    for link in links:
        for mesh in link.iter("mesh"):
            fn = mesh.get("filename")
            if fn is None:
                continue
            rel = fn[len("package://") :] if fn.startswith("package://") else fn
            prefixes.add(rel.split("/")[0])
    if len(prefixes) != 1:
        raise SystemExit(f"expected one mesh dir prefix, got {sorted(prefixes)}")
    return prefixes.pop()


def _strip_package_scheme(root: ET.Element) -> None:
    """Drop any ``package://`` scheme so paths stay plain-relative to the URDF dir."""
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("package://"):
            mesh.set("filename", fn[len("package://") :])


def _symlink(target: str, link_path: str) -> None:
    """Create/refresh a symlink link_path -> target (absolute)."""
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(os.path.abspath(target), link_path)


def _adjacent_pairs(root: ET.Element) -> list[tuple[str, str]]:
    """parent<->child link pair for every joint (always-colliding for a serial chain)."""
    pairs = []
    for j in root.findall("joint"):
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            pairs.append((p.get("link"), c.get("link")))
    return pairs


def build(xarm7_urdf: str, xarm6_robotiq_urdf: str, out_urdf: str) -> None:
    if not os.path.exists(xarm6_robotiq_urdf):
        raise SystemExit(
            f"missing {xarm6_robotiq_urdf}\n"
            "run: python -m mani_skill.utils.download_asset xarm6"
        )

    arm_dir = os.path.dirname(os.path.abspath(xarm7_urdf))
    grip_dir = os.path.dirname(os.path.abspath(xarm6_robotiq_urdf))

    arm_root = ET.parse(xarm7_urdf).getroot()
    grip_root = ET.parse(xarm6_robotiq_urdf).getroot()
    _strip_package_scheme(arm_root)
    _strip_package_scheme(grip_root)

    grip_links = {l.get("name"): l for l in grip_root.findall("link")}
    grip_joints = {j.get("name"): j for j in grip_root.findall("joint")}

    # Mesh prefixes must differ so both trees can be symlinked side by side. The grip
    # prefix is scanned over the transplanted subtree only — xArm6's own arm meshes
    # ('visual'/'collision') are left behind and must not leak in.
    arm_prefix = _mesh_prefix(arm_root.findall("link"))                       # 'meshes'
    grip_prefix = _mesh_prefix([grip_links[n] for n in GRIPPER_LINKS if n in grip_links])  # 'meshes_robotiq'
    if arm_prefix == grip_prefix:
        raise SystemExit(f"arm and gripper share mesh prefix '{arm_prefix}'; cannot graft")

    out = ET.Element("robot", {"name": "xarm7_robotiq"})

    # 1) Full xArm7 arm (link_base … link_eef, joint1..7, joint_eef).
    arm_link_names = {l.get("name") for l in arm_root.findall("link")}
    for link in arm_root.findall("link"):
        out.append(copy.deepcopy(link))
    for joint in arm_root.findall("joint"):
        out.append(copy.deepcopy(joint))

    # 2) Robotiq 2F-85 subtree.
    for name in GRIPPER_LINKS:
        if name not in grip_links:
            raise SystemExit(f"gripper link '{name}' not found in {xarm6_robotiq_urdf}")
        if name in arm_link_names:
            raise SystemExit(f"link name clash between arm and gripper: '{name}'")
        out.append(copy.deepcopy(grip_links[name]))
    for name in GRIPPER_JOINTS:
        if name not in grip_joints:
            raise SystemExit(f"gripper joint '{name}' not found in {xarm6_robotiq_urdf}")
        j = copy.deepcopy(grip_joints[name])
        if name == "gripper_fix":
            # Reparent flange: xArm6 link6 -> xArm7 link_eef (transform stays identity).
            j.find("parent").set("link", ARM_FLANGE_LINK)
        out.append(j)

    ET.indent(out, space="  ")
    out_dir = Path(out_urdf).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out).write(out_urdf, encoding="utf-8", xml_declaration=True)
    print(f"wrote URDF: {out_urdf}")

    # Symlink both mesh trees beside the URDF so the original relative paths resolve
    # for SAPIEN *and* mplib.
    _symlink(os.path.join(arm_dir, arm_prefix), str(out_dir / arm_prefix))
    _symlink(os.path.join(grip_dir, grip_prefix), str(out_dir / grip_prefix))
    print(f"linked meshes: {arm_prefix} -> {os.path.join(arm_dir, arm_prefix)}")
    print(f"linked meshes: {grip_prefix} -> {os.path.join(grip_dir, grip_prefix)}")

    _write_srdf(arm_root, grip_dir, xarm6_robotiq_urdf, out_urdf)


def _write_srdf(arm_root: ET.Element, grip_dir: str, xarm6_robotiq_urdf: str, out_urdf: str) -> None:
    """Emit an SRDF: arm-adjacent pairs + Robotiq self-collision pairs from xarm6.

    The Robotiq four-bar fingers are geometrically interlocked, so mplib must be told
    which gripper pairs never matter — otherwise it flags the gripper as permanently
    self-colliding. We lift xarm6_robotiq.srdf's gripper block verbatim, remapping its
    ``link6`` (xArm6 flange) references to xArm7's ``link_eef`` + ``link7``.
    """
    srdf_out = out_urdf.replace(".urdf", ".srdf")
    lines = ['<?xml version="1.0"?>', '<robot name="xarm7_robotiq">']

    seen: set[frozenset[str]] = set()

    def add(a: str, b: str, reason: str) -> None:
        key = frozenset((a, b))
        if a != b and key not in seen:
            seen.add(key)
            lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>')

    # Arm adjacency (serial chain).
    for a, b in _adjacent_pairs(arm_root):
        add(a, b, "Adjacent")

    # Gripper pairs from xarm6_robotiq.srdf, with link6 -> {link_eef, link7}.
    srdf6 = xarm6_robotiq_urdf.replace(".urdf", ".srdf")
    if os.path.exists(srdf6):
        s6 = ET.parse(srdf6).getroot()
        for dc in s6.findall("disable_collisions"):
            l1, l2, reason = dc.get("link1"), dc.get("link2"), dc.get("reason", "Never")
            remap = {"link6": [ARM_FLANGE_LINK, "link7"]}
            l1s = remap.get(l1, [l1])
            l2s = remap.get(l2, [l2])
            for a in l1s:
                for b in l2s:
                    add(a, b, reason)

    lines.append("</robot>")
    Path(srdf_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote SRDF: {srdf_out}")


def main(argv: list[str] | None = None) -> int:
    try:
        from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
    except Exception as exc:  # pragma: no cover - import guard
        print(f"ManiSkill import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    default_out = Path(__file__).resolve().parents[1] / "pg3d/envs/xarm_adapter/assets/xarm7_robotiq.urdf"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xarm7-urdf", default=f"{PACKAGE_ASSET_DIR}/robots/xarm7/xarm7.urdf")
    p.add_argument("--xarm6-robotiq-urdf", default=f"{ASSET_DIR}/robots/xarm6/xarm6_robotiq.urdf")
    p.add_argument("--out", default=str(default_out))
    args = p.parse_args(argv)

    build(args.xarm7_urdf, args.xarm6_robotiq_urdf, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
