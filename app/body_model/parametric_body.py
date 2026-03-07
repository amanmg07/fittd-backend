"""
Parametric human body mesh generator.

Creates a human-shaped mesh from body measurements without requiring
external model files (SMPL-X). Uses anatomical proportions to place
cross-sectional ellipses along the body and skin them into a mesh.
"""

import numpy as np
import trimesh


def ellipse_ring(center: np.ndarray, rx: float, rz: float, n: int = 24) -> np.ndarray:
    """Generate a ring of vertices forming an ellipse at a given center."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.zeros((n, 3))
    verts[:, 0] = center[0] + rx * np.cos(angles)
    verts[:, 1] = center[1]
    verts[:, 2] = center[2] + rz * np.sin(angles)
    return verts


def skin_rings(rings: list[np.ndarray], cap_top: bool = True, cap_bottom: bool = True):
    """
    Connect a sequence of vertex rings into a mesh with quads (as triangles).
    Returns (vertices, faces).
    """
    n = len(rings[0])  # verts per ring
    all_verts = np.vstack(rings)
    faces = []

    for r in range(len(rings) - 1):
        for s in range(n):
            s_next = (s + 1) % n
            v0 = r * n + s
            v1 = r * n + s_next
            v2 = (r + 1) * n + s
            v3 = (r + 1) * n + s_next
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Cap top
    if cap_top:
        top_center_idx = len(all_verts)
        top_center = rings[0].mean(axis=0)
        all_verts = np.vstack([all_verts, [top_center]])
        for s in range(n):
            s_next = (s + 1) % n
            faces.append([top_center_idx, s, s_next])

    # Cap bottom
    if cap_bottom:
        bottom_center_idx = len(all_verts)
        bottom_center = rings[-1].mean(axis=0)
        all_verts = np.vstack([all_verts, [bottom_center]])
        last_ring_start = (len(rings) - 1) * n
        for s in range(n):
            s_next = (s + 1) % n
            faces.append([bottom_center_idx, last_ring_start + s_next, last_ring_start + s])

    return all_verts, np.array(faces)


def build_body_mesh(
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hips_cm: float,
    shoulder_width_cm: float,
    neck_cm: float,
    arm_length_cm: float,
    torso_length_cm: float,
    weight_kg: float,
    gender: str = "male",
) -> trimesh.Trimesh:
    """
    Build a human-shaped mesh from body measurements.
    All inputs in cm, output mesh in meters.
    """
    # Convert to meters
    H = height_cm / 100.0
    seg = 24  # vertices per ring

    # Key heights (y coordinates, 0 = feet, H = top of head)
    ankle_y = H * 0.05
    knee_y = H * 0.27
    crotch_y = H * 0.45
    hip_y = H * 0.50
    waist_y = H * 0.58
    chest_y = H * 0.70
    shoulder_y = H * 0.78
    neck_base_y = H * 0.82
    neck_top_y = H * 0.87
    head_y = H * 0.94
    top_y = H

    # Radii from circumferences (C = pi*(a+b) approximation for ellipses)
    # Front-to-side ratio differs by region
    chest_rx = (chest_cm / 100) / (2 * np.pi) * 1.15  # wider front
    chest_rz = (chest_cm / 100) / (2 * np.pi) * 0.85  # shallower side
    waist_rx = (waist_cm / 100) / (2 * np.pi) * 1.1
    waist_rz = (waist_cm / 100) / (2 * np.pi) * 0.9
    hip_rx = (hips_cm / 100) / (2 * np.pi) * 1.1
    hip_rz = (hips_cm / 100) / (2 * np.pi) * 0.9
    shoulder_rx = (shoulder_width_cm / 100) / 2
    shoulder_rz = chest_rz * 0.85
    neck_r = (neck_cm / 100) / (2 * np.pi)

    # Head radius
    head_r = H * 0.045

    # Leg/arm thickness based on build
    bmi = weight_kg / (H * H)
    build_factor = np.clip(bmi / 22.0, 0.8, 1.4)

    thigh_rx = hip_rx * 0.42 * build_factor
    thigh_rz = hip_rz * 0.42 * build_factor
    knee_rx = thigh_rx * 0.75
    knee_rz = thigh_rz * 0.75
    calf_rx = knee_rx * 0.85
    calf_rz = knee_rz * 0.85
    ankle_rx = knee_rx * 0.55
    ankle_rz = knee_rz * 0.55

    upper_arm_r = chest_rx * 0.22 * build_factor
    forearm_r = upper_arm_r * 0.75
    wrist_r = upper_arm_r * 0.5

    meshes = []

    # ── TORSO ──
    torso_rings = []
    # Build torso profile from crotch to neck
    torso_profile = [
        (crotch_y, hip_rx * 0.85, hip_rz * 0.85),
        (hip_y,    hip_rx,        hip_rz),
        (hip_y + (waist_y - hip_y) * 0.5, (hip_rx + waist_rx) / 2, (hip_rz + waist_rz) / 2),
        (waist_y,  waist_rx,      waist_rz),
        (waist_y + (chest_y - waist_y) * 0.5, (waist_rx + chest_rx) / 2, (waist_rz + chest_rz) / 2),
        (chest_y,  chest_rx,      chest_rz),
        (chest_y + (shoulder_y - chest_y) * 0.5, (chest_rx + shoulder_rx) / 2, (chest_rz + shoulder_rz) / 2),
        (shoulder_y, shoulder_rx,  shoulder_rz),
        (neck_base_y, neck_r * 1.3, neck_r * 1.2),
    ]

    for y, rx, rz in torso_profile:
        center = np.array([0, y, 0])
        torso_rings.append(ellipse_ring(center, rx, rz, seg))

    tv, tf = skin_rings(torso_rings, cap_top=False, cap_bottom=False)
    meshes.append(trimesh.Trimesh(vertices=tv, faces=tf, process=False))

    # ── NECK ──
    neck_rings = [
        ellipse_ring(np.array([0, neck_base_y, 0]), neck_r * 1.2, neck_r * 1.1, seg),
        ellipse_ring(np.array([0, (neck_base_y + neck_top_y) / 2, 0]), neck_r, neck_r, seg),
        ellipse_ring(np.array([0, neck_top_y, 0]), neck_r * 0.95, neck_r * 0.95, seg),
    ]
    nv, nf = skin_rings(neck_rings, cap_top=False, cap_bottom=False)
    meshes.append(trimesh.Trimesh(vertices=nv, faces=nf, process=False))

    # ── HEAD ──
    head_mesh = trimesh.creation.icosphere(subdivisions=2, radius=head_r)
    head_mesh.apply_translation([0, head_y, 0])
    # Slightly elongate vertically
    head_mesh.vertices[:, 1] *= 1.2
    meshes.append(head_mesh)

    # ── LEGS ──
    leg_sep = hip_rx * 0.45  # distance between leg centers

    for side in [-1, 1]:
        cx = side * leg_sep
        leg_rings = [
            ellipse_ring(np.array([cx, crotch_y, 0]), thigh_rx, thigh_rz, seg),
            ellipse_ring(np.array([cx, crotch_y - (crotch_y - knee_y) * 0.33, 0]),
                        thigh_rx * 0.92, thigh_rz * 0.92, seg),
            ellipse_ring(np.array([cx, crotch_y - (crotch_y - knee_y) * 0.66, 0]),
                        (thigh_rx + knee_rx) / 2, (thigh_rz + knee_rz) / 2, seg),
            ellipse_ring(np.array([cx, knee_y, 0]), knee_rx, knee_rz, seg),
            ellipse_ring(np.array([cx, knee_y - (knee_y - ankle_y) * 0.33, 0]),
                        calf_rx, calf_rz, seg),
            ellipse_ring(np.array([cx, knee_y - (knee_y - ankle_y) * 0.66, 0]),
                        (calf_rx + ankle_rx) / 2, (calf_rz + ankle_rz) / 2, seg),
            ellipse_ring(np.array([cx, ankle_y, 0]), ankle_rx, ankle_rz, seg),
            ellipse_ring(np.array([cx, 0, 0]), ankle_rx * 0.9, ankle_rz * 1.3, seg),  # foot
        ]
        lv, lf = skin_rings(leg_rings, cap_top=False, cap_bottom=True)
        meshes.append(trimesh.Trimesh(vertices=lv, faces=lf, process=False))

    # ── ARMS ──
    arm_len_m = arm_length_cm / 100.0

    for side in [-1, 1]:
        sx = side * (shoulder_rx + 0.01)
        sy = shoulder_y

        # Arm hangs down with slight angle outward
        arm_dir_x = side * 0.15
        arm_dir_y = -1.0
        arm_len_norm = np.sqrt(arm_dir_x**2 + arm_dir_y**2)
        arm_dir_x /= arm_len_norm
        arm_dir_y /= arm_len_norm

        elbow_t = 0.47  # elbow at ~47% of arm length
        wrist_t = 0.95

        points = []
        for t in [0, 0.2, elbow_t, 0.65, wrist_t, 1.0]:
            px = sx + arm_dir_x * arm_len_m * t
            py = sy + arm_dir_y * arm_len_m * t
            points.append((px, py))

        radii = [
            upper_arm_r * 1.1,   # shoulder joint
            upper_arm_r,          # upper arm
            upper_arm_r * 0.85,   # elbow
            forearm_r,            # forearm
            wrist_r,              # wrist
            wrist_r * 0.7,        # hand
        ]

        arm_rings = []
        for (px, py), r in zip(points, radii):
            arm_rings.append(ellipse_ring(np.array([px, py, 0]), r, r, seg))

        av, af = skin_rings(arm_rings, cap_top=False, cap_bottom=True)
        meshes.append(trimesh.Trimesh(vertices=av, faces=af, process=False))

    # Combine all parts
    combined = trimesh.util.concatenate(meshes)
    combined.fix_normals()

    # Apply a skin-like color
    if gender == "male":
        combined.visual.face_colors = [210, 180, 150, 255]
    else:
        combined.visual.face_colors = [220, 190, 160, 255]

    return combined
