"""mplib motion planner for the no-gripper xArm7.

ManiSkill's ``PandaArmMotionPlanningSolver`` and ``XArm6RobotiqMotionPlanningSolver``
both extend ``TwoFingerGripperMotionPlanningSolver`` (gripper open/close logic +
8-dim action with a gripper column). The xArm7 target has no gripper, so we extend
the gripper-agnostic ``BaseMotionPlanningSolver`` directly. That base is pure mplib:
``setup_planner`` builds an ``mplib.Planner`` from the env agent's URDF, link/joint
names and ``MOVE_GROUP`` — nothing Panda- or gripper-specific.

Only override: the bundled ``xarm7.urdf`` ships without a sibling ``.srdf``, and the
base ``setup_planner`` assumes ``<urdf>.srdf`` exists. We pass an empty SRDF when it
is missing so mplib just disables SRDF-based self-collision pairs (planning still
respects URDF collision geometry).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import mplib
import trimesh
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import (
    BaseMotionPlanningSolver,
)


def _watertight_convex_hull(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Convex hull guaranteed watertight (every edge shared by exactly two faces).

    trimesh's ``Trimesh.convex_hull`` post-merges coplanar faces; on near-coplanar
    input (e.g. the xArm gripper's inner-knuckle links) that merge drops triangles
    and leaves open edges. SAPIEN/mplib runs a strict watertight validator on each
    convex collision mesh at ``ArticulatedModel`` construction and hard-errors on
    those open edges ("The mesh is not watertight"). We instead build the hull from
    scipy's raw simplical output with ``process=False`` (no vertex merge), so the
    triangulation stays a clean closed manifold.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    hull = ConvexHull(pts)
    used = np.unique(hull.simplices)
    remap = {int(old): i for i, old in enumerate(used)}
    faces = np.array([[remap[int(v)] for v in simplex] for simplex in hull.simplices])
    out = trimesh.Trimesh(vertices=pts[used], faces=faces, process=False)
    trimesh.repair.fix_normals(out)
    return out


def _ensure_convex_collision_meshes(urdf_path: str) -> None:
    """Generate/repair the ``<mesh>.convex.stl`` files mplib expects.

    mplib (0.1.x) always loads ``<collision_mesh>.convex.stl`` for each link, but the
    bundled ``xarm7.urdf`` ships only raw ``.obj``/``.stl`` collision meshes (Panda
    ships the pre-baked ``.convex.stl`` hulls). We compute a convex hull per collision
    mesh once and cache it next to the source mesh.

    Self-healing: a missing hull is generated, and an existing hull that is *not*
    watertight is regenerated. The latter matters because earlier hulls baked with
    ``trimesh.convex_hull`` can be non-manifold (see :func:`_watertight_convex_hull`),
    which crashes mplib. Idempotent: a present, watertight hull is left untouched.
    """
    urdf_dir = os.path.dirname(urdf_path)
    root = ET.parse(urdf_path).getroot()
    seen: set[str] = set()
    for collision in root.iter("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        rel = mesh.get("filename")
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        src = os.path.normpath(os.path.join(urdf_dir, rel))
        dst = f"{src}.convex.stl"
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and trimesh.load(dst, force="mesh").is_watertight:
            continue
        hull = _watertight_convex_hull(trimesh.load(src, force="mesh"))
        hull.export(dst)


def _gripper_rigid_cluster(root: ET.Element) -> list[str]:
    """Links that share a rigid body with the gripper, hence permanently inter-collide.

    Two families, both always-colliding at *every* configuration:

    * the **gripper subtree** — the child of ``gripper_fix`` and all its descendants
      (the whole passive gripper);
    * the **fixed-joint ancestors** — arm links reachable upward from the gripper base
      through *fixed* joints only (e.g. ``link_eef`` and ``link7``, bolted on via the
      fixed ``gripper_fix`` / ``joint_eef``). These never move relative to the gripper
      base, so e.g. ``link7`` ↔ ``xarm_gripper_base_link`` overlaps forever and mplib
      would otherwise flag the robot as permanently self-colliding — IK then returns no
      valid solution and every plan fails.

    Returns ``[]`` for a no-gripper arm (no ``gripper_fix`` joint). All pairs among the
    returned links are disabled in the SRDF (see :func:`_ensure_srdf`).
    """
    joints = root.findall("joint")
    fix = next((j for j in joints if j.get("name") == "gripper_fix"), None)
    if fix is None or fix.find("child") is None:
        return []
    base = fix.find("child").get("link")

    children: dict[str, list[str]] = {}
    parent_joint: dict[str, ET.Element] = {}
    for j in joints:
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            children.setdefault(p.get("link"), []).append(c.get("link"))
            parent_joint[c.get("link")] = j

    cluster: list[str] = []
    stack = [base]
    while stack:  # whole gripper subtree, downward
        link = stack.pop()
        cluster.append(link)
        stack.extend(children.get(link, []))

    link = base  # walk upward through fixed joints only
    while link in parent_joint and parent_joint[link].get("type") == "fixed":
        link = parent_joint[link].find("parent").get("link")
        if link not in cluster:
            cluster.append(link)

    return cluster


def _ensure_srdf(urdf_path: str) -> str:
    """Generate/repair an SRDF disabling the self-collision pairs mplib must ignore.

    Two pair families are always-colliding and must be declared, else mplib (0.1.x)
    auto-detects them at construction — slow, and it crashes on ``link_name_2_idx``:

    * **Adjacent pairs** — links joined by a URDF joint (a serial arm's only
      always-colliding pairs).
    * **Gripper rigid-cluster pairs** — for a *passive* gripper held open (our reach
      task), the whole gripper is a rigid blob, and its closed four-bar linkage puts
      non-adjacent links (e.g. ``*_inner_knuckle`` ↔ ``*_finger``) permanently in
      contact; additionally the gripper base is bolted to the wrist through fixed
      joints, so arm links like ``link7`` overlap it forever. We disable *every* pair
      among the gripper rigid cluster (see :func:`_gripper_rigid_cluster`). Collisions
      of the arm/gripper with earlier links and the environment stay checked, so
      planning is still safe.

    Self-healing: an existing SRDF (e.g. ManiSkill's hand-written adjacent-only one,
    or the repo's Robotiq SRDF) is reused when it already covers every required pair,
    and otherwise rewritten as the union of its pairs and the required set — so a
    stale adjacent-only SRDF gets upgraded in place rather than silently failing.
    Returns the SRDF path.
    """
    srdf_path = urdf_path.replace(".urdf", ".srdf")
    root = ET.parse(urdf_path).getroot()
    robot_name = root.get("name", "robot")

    required: dict[frozenset[str], str] = {}

    def need(a: str, b: str, reason: str) -> None:
        if a and b and a != b:
            required.setdefault(frozenset((a, b)), reason)

    for j in root.findall("joint"):
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            need(p.get("link"), c.get("link"), "Adjacent")

    grip = _gripper_rigid_cluster(root)
    for i in range(len(grip)):
        for k in range(i + 1, len(grip)):
            need(grip[i], grip[k], "Gripper")

    existing: dict[frozenset[str], str] = {}
    if os.path.exists(srdf_path):
        for dc in ET.parse(srdf_path).getroot().findall("disable_collisions"):
            l1, l2 = dc.get("link1"), dc.get("link2")
            if l1 and l2:
                existing[frozenset((l1, l2))] = dc.get("reason", "Never")
        if all(key in existing for key in required):
            return srdf_path  # already covers everything we need

    merged = {**required, **existing}  # keep any extra hand-tuned pairs too
    lines = [f'<robot name="{robot_name}">']
    for key, reason in merged.items():
        a, b = tuple(key) if len(key) == 2 else (next(iter(key)), next(iter(key)))
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>')
    lines.append("</robot>\n")
    with open(srdf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srdf_path


class XArm7MotionPlanningSolverBase(BaseMotionPlanningSolver):
    """Shared setup logic for both xArm7 planner variants."""

    def setup_planner(self):
        urdf = self.env_agent.urdf_path
        _ensure_convex_collision_meshes(urdf)
        srdf = _ensure_srdf(urdf)
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner


class XArm7NoGripperMotionPlanningSolver(XArm7MotionPlanningSolverBase):
    """mplib screw/RRTConnect planner for ``xarm7_nogripper`` (TCP = ``link_eef``)."""

    MOVE_GROUP = "link_eef"

    def __init__(self, *args, visualize_target_grasp_pose: bool = False, **kwargs):
        # Accept and ignore visualize_target_grasp_pose so the shared data-gen
        # call site works for both no-gripper and Panda/gripper planners.
        super().__init__(*args, **kwargs)


class XArm7GripperMotionPlanningSolver(XArm7MotionPlanningSolverBase):
    """mplib planner for ``xarm7_gripper`` (TCP = ``link_tcp``, 172 mm past flange).

    MOVE_GROUP resolves to ``link_tcp`` — the kinematic chain to that link passes
    through joint1-7 only (the gripper joints are on side branches not in the
    chain, and gripper_fix / joint_tcp are fixed joints transparent to mplib).
    mplib therefore plans over 7 DOFs and returns 7-dim position waypoints.

    The passive gripper controller adds no action dimensions, so ``env.step``
    also expects 7-dim actions. ``follow_path`` inherits from the base unchanged.
    """

    MOVE_GROUP = "link_tcp"

    def __init__(self, *args, visualize_target_grasp_pose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)


class XArm7RobotiqMotionPlanningSolver(XArm7MotionPlanningSolverBase):
    """mplib planner for ``xarm7_robotiq`` (TCP = ``eef``, 150 mm past the flange).

    Same 7-DOF contract as the xArm-gripper planner: the chain to ``eef`` runs
    through joint1-7 only (the Robotiq finger joints are side branches; gripper_fix
    and eef_joint are fixed), so mplib plans 7 DOFs and returns 7-dim waypoints, and
    the passive gripper adds no action dims. The combined URDF ships a sibling SRDF
    (Robotiq self-collision pairs) and pre-baked ``.convex.stl`` hulls, so the base
    ``setup_planner`` finds both and skips regeneration.
    """

    MOVE_GROUP = "eef"

    def __init__(self, *args, visualize_target_grasp_pose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
