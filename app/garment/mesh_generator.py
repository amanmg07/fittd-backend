"""
Garment mesh generator: creates a 3D garment mesh that can be
draped over a body model.

Approach:
1. Start from a template garment mesh (parametric t-shirt, etc.)
2. Scale it to match the selected garment size dimensions
3. Simulate cloth draping onto the body mesh
4. Export as GLB for the mobile viewer
"""

import numpy as np
import trimesh


def create_tshirt_template(
    chest_cm: float,
    length_cm: float,
    shoulder_cm: float,
    sleeve_cm: float | None = None,
) -> trimesh.Trimesh:
    """
    Create a parametric t-shirt mesh from dimensions.

    The mesh is built from simple geometric primitives
    scaled to match the garment measurements.
    """
    # Convert cm to meters for the 3D scene
    chest = chest_cm / 100.0
    length = length_cm / 100.0
    shoulder = shoulder_cm / 100.0
    sleeve = (sleeve_cm or 25.0) / 100.0

    half_chest = chest / (2 * np.pi)  # radius from circumference
    half_length = length / 2
    half_shoulder = shoulder / 2

    # Build torso as a tapered cylinder
    n_segments = 32
    n_rows = 10

    vertices = []
    faces = []

    for row in range(n_rows + 1):
        t = row / n_rows  # 0 = top (shoulders), 1 = bottom (hem)
        y = half_length - t * length

        # Taper: wider at chest (row ~3), narrower at waist (row ~7)
        chest_factor = 1.0 + 0.08 * np.sin(np.pi * t * 0.8)
        radius = half_chest * chest_factor

        for seg in range(n_segments):
            angle = 2 * np.pi * seg / n_segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            vertices.append([x, y, z])

    # Build faces
    for row in range(n_rows):
        for seg in range(n_segments):
            next_seg = (seg + 1) % n_segments
            v0 = row * n_segments + seg
            v1 = row * n_segments + next_seg
            v2 = (row + 1) * n_segments + seg
            v3 = (row + 1) * n_segments + next_seg

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Add sleeve geometry (simplified as tapered cylinders)
    if sleeve_cm and sleeve_cm > 5:
        for side in [-1, 1]:
            sleeve_base_y = half_length - length * 0.15  # shoulder height
            sleeve_n_rows = 5
            sleeve_n_segs = 16
            sleeve_radius_start = half_chest * 0.28
            sleeve_radius_end = half_chest * 0.22

            base_idx = len(vertices)

            for row in range(sleeve_n_rows + 1):
                t = row / sleeve_n_rows
                r = sleeve_radius_start + t * (sleeve_radius_end - sleeve_radius_start)
                x_offset = side * (half_shoulder + t * sleeve * 0.8)
                y = sleeve_base_y - t * sleeve * 0.3

                for seg in range(sleeve_n_segs):
                    angle = 2 * np.pi * seg / sleeve_n_segs
                    x = x_offset + r * np.cos(angle) * 0.3
                    z = r * np.sin(angle)
                    vertices.append([x, y, z])

            for row in range(sleeve_n_rows):
                for seg in range(sleeve_n_segs):
                    next_seg = (seg + 1) % sleeve_n_segs
                    v0 = base_idx + row * sleeve_n_segs + seg
                    v1 = base_idx + row * sleeve_n_segs + next_seg
                    v2 = base_idx + (row + 1) * sleeve_n_segs + seg
                    v3 = base_idx + (row + 1) * sleeve_n_segs + next_seg

                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])

    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    return mesh


def drape_garment_on_body(
    garment_mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """
    Simple garment draping: project garment vertices onto the body surface
    with an offset to simulate cloth sitting on the body.

    For production, this would use proper cloth simulation (position-based
    dynamics or XPBD), but this gives a reasonable visual approximation.
    """
    offset = 0.005  # 5mm offset from body surface

    body_proximity = trimesh.proximity.ProximityQuery(body_mesh)
    closest_points, distances, _ = body_proximity.on_surface(garment_mesh.vertices)

    # For vertices that are inside the body, push them out
    normals = body_mesh.vertex_normals
    draped_vertices = garment_mesh.vertices.copy()

    for i, (closest, dist) in enumerate(zip(closest_points, distances)):
        if dist < offset:
            # Vertex is too close to or inside body — push out
            direction = garment_mesh.vertices[i] - closest
            if np.linalg.norm(direction) > 1e-8:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([0, 0, 1])
            draped_vertices[i] = closest + direction * offset

    result = trimesh.Trimesh(vertices=draped_vertices, faces=garment_mesh.faces)
    result.fix_normals()
    return result


def generate_fit_map(
    garment_mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Generate a per-vertex color map showing tight (red) vs loose (blue) areas.
    Returns RGBA colors array (N, 4).
    """
    body_proximity = trimesh.proximity.ProximityQuery(body_mesh)
    _, distances, _ = body_proximity.on_surface(garment_mesh.vertices)

    # Normalize distances: 0 = touching body, ~0.05m = loose
    max_dist = 0.05
    normalized = np.clip(distances / max_dist, 0, 1)

    colors = np.zeros((len(distances), 4), dtype=np.uint8)
    for i, d in enumerate(normalized):
        if d < 0.2:
            # Tight: red
            colors[i] = [255, int(d * 5 * 255), 0, 200]
        elif d < 0.5:
            # Good fit: green
            t = (d - 0.2) / 0.3
            colors[i] = [int((1 - t) * 255), 255, int(t * 100), 200]
        else:
            # Loose: blue
            t = (d - 0.5) / 0.5
            colors[i] = [0, int((1 - t) * 200), int(128 + t * 127), 200]

    return colors


def build_tryon_scene(
    body_glb: bytes,
    garment_size_dims: dict,
    garment_color: str = "#333333",
) -> bytes:
    """
    Build a complete try-on scene: body + draped garment.
    Returns GLB bytes of the combined scene.
    """
    body_mesh = trimesh.load(trimesh.util.wrap_as_stream(body_glb), file_type="glb")
    if isinstance(body_mesh, trimesh.Scene):
        body_mesh = trimesh.util.concatenate(body_mesh.dump())

    garment_template = create_tshirt_template(
        chest_cm=garment_size_dims.get("chest_cm", 100),
        length_cm=garment_size_dims.get("length_cm", 72),
        shoulder_cm=garment_size_dims.get("shoulder_cm", 47),
        sleeve_cm=garment_size_dims.get("sleeve_cm", 63),
    )

    draped_garment = drape_garment_on_body(garment_template, body_mesh)

    # Apply color to garment
    fit_colors = generate_fit_map(draped_garment, body_mesh)
    draped_garment.visual.vertex_colors = fit_colors

    scene = trimesh.Scene()
    scene.add_geometry(body_mesh, node_name="body")
    scene.add_geometry(draped_garment, node_name="garment")

    return scene.export(file_type="glb")
