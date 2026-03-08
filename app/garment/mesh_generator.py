"""
Garment mesh generator: creates a 3D garment mesh with proper shape,
UV mapping, and product image texture that can be draped over a body model.

Approach:
1. Build a realistic garment shape (neckline, hem curve, sleeve shape)
2. Create UV coordinates for texture mapping
3. Download and apply the product image as texture
4. Drape onto body mesh with proximity-based fitting
5. Export as GLB with embedded texture
"""

import numpy as np
import trimesh
import io
import httpx
from PIL import Image


def create_tshirt_mesh(
    chest_cm: float,
    length_cm: float,
    shoulder_cm: float,
    sleeve_cm: float | None = None,
) -> trimesh.Trimesh:
    """
    Create a realistic t-shirt mesh with proper neckline, hem,
    and sleeve shapes. Includes UV coordinates for texturing.
    """
    chest = chest_cm / 100.0
    length = length_cm / 100.0
    shoulder = shoulder_cm / 100.0
    sleeve = (sleeve_cm or 25.0) / 100.0

    half_chest_r = chest / (2 * np.pi)
    half_shoulder = shoulder / 2

    n_seg = 40  # segments around circumference
    n_rows = 18  # rows along body length

    vertices = []
    uvs = []
    faces = []

    # Neckline shape: how much to cut in at the top
    neck_r = half_chest_r * 0.35

    for row in range(n_rows + 1):
        t = row / n_rows  # 0 = top (neckline), 1 = bottom (hem)

        # Vertical position
        top_y = length * 0.52  # above center (shoulder height)
        y = top_y - t * length

        # Body shape profile: slight tapering
        if t < 0.15:
            # Upper chest / shoulder area — wider
            shape_factor = 1.0 + 0.05 * (1 - t / 0.15)
        elif t < 0.5:
            # Chest area — full width
            shape_factor = 1.0 + 0.06 * np.sin(np.pi * (t - 0.15) / 0.35)
        elif t < 0.8:
            # Waist area — narrower
            shape_factor = 1.0 - 0.04 * np.sin(np.pi * (t - 0.5) / 0.3)
        else:
            # Hem — slight flare
            shape_factor = 1.0 + 0.02 * ((t - 0.8) / 0.2)

        radius = half_chest_r * shape_factor

        # Hem curve: slightly longer at center front/back
        if t > 0.9:
            hem_extra = 0.008 * np.sin(np.pi * (t - 0.9) / 0.1)
        else:
            hem_extra = 0

        for seg in range(n_seg):
            angle = 2 * np.pi * seg / n_seg
            u = seg / n_seg
            v = t

            x = radius * np.cos(angle)
            z = radius * np.sin(angle)

            # Neckline cutout at top rows
            if t < 0.08:
                # Distance from center-front (angle = 0 or 2pi)
                front_dist = min(abs(angle), abs(angle - 2 * np.pi))
                back_dist = abs(angle - np.pi)

                # Front neckline: deeper V/round
                front_cutout = max(0, 1 - front_dist / 0.8) * 0.06 * (1 - t / 0.08)
                # Back neckline: shallower
                back_cutout = max(0, 1 - back_dist / 0.5) * 0.025 * (1 - t / 0.08)

                y_adjusted = y + front_cutout + back_cutout
            else:
                y_adjusted = y - hem_extra * abs(np.cos(angle))

            vertices.append([x, y_adjusted, z])
            uvs.append([u, v])

    # Build faces for torso
    for row in range(n_rows):
        for seg in range(n_seg):
            next_seg = (seg + 1) % n_seg
            v0 = row * n_seg + seg
            v1 = row * n_seg + next_seg
            v2 = (row + 1) * n_seg + seg
            v3 = (row + 1) * n_seg + next_seg
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # ── Sleeves ──
    if sleeve_cm and sleeve_cm > 5:
        sleeve_n_rows = 8
        sleeve_n_segs = 24
        sleeve_radius_start = half_chest_r * 0.30
        sleeve_radius_end = half_chest_r * 0.24

        for side in [-1, 1]:
            sleeve_base_y = length * 0.52 - length * 0.06
            base_idx = len(vertices)

            # Sleeve angle: follows arm in A-pose
            sleeve_angle = np.radians(30)
            sleeve_dir_x = side * np.sin(sleeve_angle)
            sleeve_dir_y = -np.cos(sleeve_angle)

            for row in range(sleeve_n_rows + 1):
                t = row / sleeve_n_rows
                r = sleeve_radius_start + t * (sleeve_radius_end - sleeve_radius_start)

                # Position along sleeve direction
                dist = t * sleeve * 0.85
                center_x = side * half_shoulder + sleeve_dir_x * dist
                center_y = sleeve_base_y + sleeve_dir_y * dist

                # Slight taper shape
                r_shape = r * (1.0 - 0.08 * t)

                for seg in range(sleeve_n_segs):
                    angle = 2 * np.pi * seg / sleeve_n_segs

                    # Flatten sleeve cross-section slightly
                    x = center_x + r_shape * np.cos(angle) * 0.85
                    z = r_shape * np.sin(angle)
                    y = center_y + r_shape * np.cos(angle) * 0.15 * (1 - t)

                    # UV: map sleeve to a side section of the texture
                    u = 0.8 + side * 0.1 + seg / sleeve_n_segs * 0.1 * side
                    v = t * 0.3

                    vertices.append([x, y, z])
                    uvs.append([u, v])

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
    uvs = np.array(uvs)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh.fix_normals()

    return mesh


def download_garment_image(image_url: str) -> Image.Image | None:
    """Download garment product image for texturing."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as http:
            resp = http.get(image_url)
            resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception:
        return None


def create_garment_texture(product_image: Image.Image | None, color: str = "#333333") -> Image.Image:
    """
    Create a texture image for the garment mesh.
    If product image is available, use it. Otherwise, create a solid color.
    """
    tex_size = 1024

    if product_image is not None:
        # Resize product image to fill texture
        img = product_image.resize((tex_size, tex_size), Image.LANCZOS)
        # Convert to RGB
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img
    else:
        # Solid color fallback
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return Image.new("RGB", (tex_size, tex_size), (r, g, b))


def apply_texture_to_mesh(mesh: trimesh.Trimesh, texture_image: Image.Image) -> trimesh.Trimesh:
    """Apply a texture image to a mesh using its UV coordinates."""
    # Get existing UVs
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = mesh.visual.uv
    else:
        # Generate basic cylindrical UVs as fallback
        verts = mesh.vertices
        angles = np.arctan2(verts[:, 2], verts[:, 0])
        u = (angles + np.pi) / (2 * np.pi)
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        v = (verts[:, 1] - y_min) / (y_max - y_min + 1e-8)
        uv = np.column_stack([u, v])

    # Convert PIL image to trimesh material
    img_bytes = io.BytesIO()
    texture_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.open(img_bytes),
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


def drape_garment_on_body(
    garment_mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """
    Drape garment onto body: push vertices that are inside or too close
    to the body surface outward, preserving the garment's shape where possible.
    """
    offset = 0.006  # 6mm offset from body surface

    body_proximity = trimesh.proximity.ProximityQuery(body_mesh)
    closest_points, distances, _ = body_proximity.on_surface(garment_mesh.vertices)

    draped_vertices = garment_mesh.vertices.copy()

    for i, (closest, dist) in enumerate(zip(closest_points, distances)):
        if dist < offset:
            direction = garment_mesh.vertices[i] - closest
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])
            draped_vertices[i] = closest + direction * offset

    result = trimesh.Trimesh(
        vertices=draped_vertices,
        faces=garment_mesh.faces,
        process=False,
    )

    # Preserve UV and texture from original
    if hasattr(garment_mesh.visual, 'uv') and garment_mesh.visual.uv is not None:
        result.visual = garment_mesh.visual

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

    max_dist = 0.05
    normalized = np.clip(distances / max_dist, 0, 1)

    colors = np.zeros((len(distances), 4), dtype=np.uint8)
    for i, d in enumerate(normalized):
        if d < 0.2:
            colors[i] = [255, int(d * 5 * 255), 0, 200]
        elif d < 0.5:
            t = (d - 0.2) / 0.3
            colors[i] = [int((1 - t) * 255), 255, int(t * 100), 200]
        else:
            t = (d - 0.5) / 0.5
            colors[i] = [0, int((1 - t) * 200), int(128 + t * 127), 200]

    return colors


def build_tryon_scene(
    body_glb: bytes,
    garment_size_dims: dict,
    garment_color: str = "#333333",
    garment_image_url: str | None = None,
) -> bytes:
    """
    Build a complete try-on scene: body + draped garment with texture.
    Returns GLB bytes of the combined scene.
    """
    body_mesh = trimesh.load(trimesh.util.wrap_as_stream(body_glb), file_type="glb")
    if isinstance(body_mesh, trimesh.Scene):
        body_mesh = trimesh.util.concatenate(body_mesh.dump())

    # Create garment with proper shape
    garment_template = create_tshirt_mesh(
        chest_cm=garment_size_dims.get("chest_cm", 100),
        length_cm=garment_size_dims.get("length_cm", 72),
        shoulder_cm=garment_size_dims.get("shoulder_cm", 47),
        sleeve_cm=garment_size_dims.get("sleeve_cm", 63),
    )

    # Position garment on body (align waist height)
    body_center_y = (body_mesh.vertices[:, 1].min() + body_mesh.vertices[:, 1].max()) / 2
    garment_center_y = (garment_template.vertices[:, 1].min() + garment_template.vertices[:, 1].max()) / 2
    garment_template.vertices[:, 1] += (body_center_y - garment_center_y) * 0.15

    # Drape garment on body
    draped_garment = drape_garment_on_body(garment_template, body_mesh)

    # Apply texture
    product_image = None
    if garment_image_url:
        product_image = download_garment_image(garment_image_url)

    texture = create_garment_texture(product_image, garment_color)
    draped_garment = apply_texture_to_mesh(draped_garment, texture)

    # Build scene
    scene = trimesh.Scene()
    scene.add_geometry(body_mesh, node_name="body")
    scene.add_geometry(draped_garment, node_name="garment")

    return scene.export(file_type="glb")
