"""
Parametric human body mesh generator.

Creates a realistic human-shaped mesh from body measurements.
Uses dense cross-sectional profiles with cubic interpolation
and subdivision smoothing for a natural appearance.
"""

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline


def ellipse_ring(center: np.ndarray, rx: float, rz: float, n: int = 48) -> np.ndarray:
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
    n = len(rings[0])
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

    if cap_top:
        top_center_idx = len(all_verts)
        top_center = rings[0].mean(axis=0)
        all_verts = np.vstack([all_verts, [top_center]])
        for s in range(n):
            s_next = (s + 1) % n
            faces.append([top_center_idx, s, s_next])

    if cap_bottom:
        bottom_center_idx = len(all_verts)
        bottom_center = rings[-1].mean(axis=0)
        all_verts = np.vstack([all_verts, [bottom_center]])
        last_ring_start = (len(rings) - 1) * n
        for s in range(n):
            s_next = (s + 1) % n
            faces.append([bottom_center_idx, last_ring_start + s_next, last_ring_start + s])

    return all_verts, np.array(faces)


def interpolate_profile(key_points: list[tuple], num_rings: int = 30) -> list[tuple]:
    """
    Smoothly interpolate between key body profile points using cubic splines.
    key_points: list of (y, rx, rz) tuples
    Returns densely sampled (y, rx, rz) tuples.
    """
    key_points = sorted(key_points, key=lambda p: p[0])
    ys = [p[0] for p in key_points]
    rxs = [p[1] for p in key_points]
    rzs = [p[2] for p in key_points]

    cs_rx = CubicSpline(ys, rxs, bc_type='clamped')
    cs_rz = CubicSpline(ys, rzs, bc_type='clamped')

    y_dense = np.linspace(ys[0], ys[-1], num_rings)
    result = []
    for y in y_dense:
        rx = max(0.001, float(cs_rx(y)))
        rz = max(0.001, float(cs_rz(y)))
        result.append((float(y), rx, rz))

    return result


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
    Build a realistic human-shaped mesh from body measurements.
    All inputs in cm, output mesh in meters.

    Uses 48 vertices per ring, cubic spline interpolation between
    anatomical landmarks, and gender-specific proportions.
    """
    H = height_cm / 100.0
    seg = 48  # vertices per ring — double the old 24 for smoothness

    # ── Key heights (y coordinates, 0 = feet, H = top of head) ──
    foot_y = 0.0
    ankle_y = H * 0.05
    mid_calf_y = H * 0.16
    knee_y = H * 0.27
    mid_thigh_y = H * 0.36
    crotch_y = H * 0.45
    hip_y = H * 0.50
    waist_y = H * 0.58
    lower_chest_y = H * 0.65
    chest_y = H * 0.70
    upper_chest_y = H * 0.74
    shoulder_y = H * 0.78
    neck_base_y = H * 0.82
    neck_mid_y = H * 0.85
    neck_top_y = H * 0.87
    chin_y = H * 0.88
    head_center_y = H * 0.92
    head_top_y = H * 0.97

    # ── Radii from circumferences ──
    # Use elliptical cross-sections: front wider, side shallower
    # Gender-specific front-to-side ratios
    if gender == "female":
        chest_front_ratio, chest_side_ratio = 1.10, 0.90
        waist_front_ratio, waist_side_ratio = 1.08, 0.92
        hip_front_ratio, hip_side_ratio = 1.15, 0.85
    else:
        chest_front_ratio, chest_side_ratio = 1.15, 0.85
        waist_front_ratio, waist_side_ratio = 1.10, 0.90
        hip_front_ratio, hip_side_ratio = 1.10, 0.90

    chest_rx = (chest_cm / 100) / (2 * np.pi) * chest_front_ratio
    chest_rz = (chest_cm / 100) / (2 * np.pi) * chest_side_ratio
    waist_rx = (waist_cm / 100) / (2 * np.pi) * waist_front_ratio
    waist_rz = (waist_cm / 100) / (2 * np.pi) * waist_side_ratio
    hip_rx = (hips_cm / 100) / (2 * np.pi) * hip_front_ratio
    hip_rz = (hips_cm / 100) / (2 * np.pi) * hip_side_ratio
    shoulder_rx = (shoulder_width_cm / 100) / 2
    shoulder_rz = chest_rz * 0.82
    neck_r = (neck_cm / 100) / (2 * np.pi)

    # Head dimensions
    head_rx = H * 0.046
    head_rz = H * 0.042

    # Build/BMI factor for limb thickness
    bmi = weight_kg / (H * H)
    build_factor = np.clip(bmi / 22.0, 0.8, 1.4)

    # Limb radii
    thigh_rx = hip_rx * 0.44 * build_factor
    thigh_rz = hip_rz * 0.44 * build_factor
    knee_rx = thigh_rx * 0.72
    knee_rz = thigh_rz * 0.72
    calf_rx = knee_rx * 0.88
    calf_rz = knee_rz * 0.88
    ankle_rx = knee_rx * 0.52
    ankle_rz = knee_rz * 0.52

    upper_arm_r = chest_rx * 0.22 * build_factor
    forearm_r = upper_arm_r * 0.72
    wrist_r = upper_arm_r * 0.48
    hand_r = wrist_r * 0.65

    meshes = []

    # ── TORSO (spline-interpolated) ──
    torso_key_points = [
        (crotch_y,      hip_rx * 0.82,  hip_rz * 0.82),
        (hip_y,         hip_rx,         hip_rz),
        (waist_y,       waist_rx,       waist_rz),
        (lower_chest_y, (waist_rx + chest_rx) / 2, (waist_rz + chest_rz) / 2),
        (chest_y,       chest_rx,       chest_rz),
        (upper_chest_y, (chest_rx + shoulder_rx) * 0.52, (chest_rz + shoulder_rz) * 0.52),
        (shoulder_y,    shoulder_rx,    shoulder_rz),
        (neck_base_y,   neck_r * 1.25,  neck_r * 1.15),
    ]

    torso_profile = interpolate_profile(torso_key_points, num_rings=28)
    torso_rings = []
    for y, rx, rz in torso_profile:
        torso_rings.append(ellipse_ring(np.array([0, y, 0]), rx, rz, seg))

    tv, tf = skin_rings(torso_rings, cap_top=False, cap_bottom=False)
    meshes.append(trimesh.Trimesh(vertices=tv, faces=tf, process=False))

    # ── NECK (spline-interpolated) ──
    neck_key = [
        (neck_base_y, neck_r * 1.20, neck_r * 1.10),
        (neck_mid_y,  neck_r * 0.98, neck_r * 0.95),
        (neck_top_y,  neck_r * 0.92, neck_r * 0.90),
        (chin_y,      neck_r * 0.85, neck_r * 0.82),
    ]
    neck_profile = interpolate_profile(neck_key, num_rings=10)
    neck_rings = []
    for y, rx, rz in neck_profile:
        neck_rings.append(ellipse_ring(np.array([0, y, 0]), rx, rz, seg))

    nv, nf = skin_rings(neck_rings, cap_top=False, cap_bottom=False)
    meshes.append(trimesh.Trimesh(vertices=nv, faces=nf, process=False))

    # ── HEAD (higher-res icosphere, elongated) ──
    head_mesh = trimesh.creation.icosphere(subdivisions=3, radius=head_rx)
    head_mesh.apply_translation([0, head_center_y, 0])
    # Elongate vertically for realistic head shape
    head_mesh.vertices[:, 1] *= 1.25
    # Slightly flatten front-to-back
    head_mesh.vertices[:, 2] *= (head_rz / head_rx)
    meshes.append(head_mesh)

    # ── LEGS (spline-interpolated) ──
    leg_sep = hip_rx * 0.44

    for side in [-1, 1]:
        cx = side * leg_sep
        leg_key = [
            (crotch_y,    thigh_rx,            thigh_rz),
            (mid_thigh_y, thigh_rx * 0.88,     thigh_rz * 0.88),
            (knee_y,      knee_rx,             knee_rz),
            (mid_calf_y,  calf_rx,             calf_rz),
            (ankle_y,     ankle_rx,            ankle_rz),
            (foot_y,      ankle_rx * 0.85,     ankle_rz * 1.35),
        ]
        leg_profile = interpolate_profile(leg_key, num_rings=18)
        leg_rings = []
        for y, rx, rz in leg_profile:
            leg_rings.append(ellipse_ring(np.array([cx, y, 0]), rx, rz, seg))

        lv, lf = skin_rings(leg_rings, cap_top=False, cap_bottom=True)
        meshes.append(trimesh.Trimesh(vertices=lv, faces=lf, process=False))

    # ── ARMS (A-pose: angled ~30° outward for garment visibility) ──
    arm_len_m = arm_length_cm / 100.0
    arm_angle = np.radians(30)  # A-pose angle from vertical

    for side_sign in [-1, 1]:
        sx = side_sign * (shoulder_rx + 0.005)
        sy = shoulder_y

        # Arm direction: angled outward and slightly down
        arm_dir_x = side_sign * np.sin(arm_angle)
        arm_dir_y = -np.cos(arm_angle)

        # Key points along arm with t = fraction of arm length
        arm_key_t = [0.0, 0.15, 0.35, 0.47, 0.60, 0.80, 0.95, 1.0]
        arm_radii = [
            upper_arm_r * 1.12,  # shoulder joint
            upper_arm_r * 1.02,  # deltoid
            upper_arm_r * 0.92,  # mid-upper-arm
            upper_arm_r * 0.82,  # elbow
            forearm_r * 1.05,    # below elbow
            forearm_r * 0.88,    # mid-forearm
            wrist_r,             # wrist
            hand_r,              # hand
        ]

        arm_rings = []
        for t, r in zip(arm_key_t, arm_radii):
            px = sx + arm_dir_x * arm_len_m * t
            py = sy + arm_dir_y * arm_len_m * t
            arm_rings.append(ellipse_ring(np.array([px, py, 0]), r, r, seg))

        av, af = skin_rings(arm_rings, cap_top=False, cap_bottom=True)
        meshes.append(trimesh.Trimesh(vertices=av, faces=af, process=False))

    # ── Combine and smooth ──
    combined = trimesh.util.concatenate(meshes)
    combined.fix_normals()

    # Apply Laplacian smoothing for a more natural surface
    trimesh.smoothing.filter_laplacian(combined, iterations=2, lamb=0.5)
    combined.fix_normals()

    # Skin-like color
    if gender == "male":
        combined.visual.face_colors = [210, 180, 150, 255]
    else:
        combined.visual.face_colors = [220, 190, 160, 255]

    return combined
