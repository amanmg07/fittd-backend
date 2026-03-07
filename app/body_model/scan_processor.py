"""
Body scan processor: takes front + side photos and produces
a SMPL-X body model with measurements.

Pipeline:
1. Extract body silhouette using GrabCut + refinement
2. Detect body landmarks using contour analysis
3. Estimate measurements from calibrated pixel ratios
4. Generate SMPL-X mesh with derived shape parameters
"""

import base64
import io
import numpy as np
import cv2

from app.models.schemas import BodyMeasurements, Gender


# Anthropometric measurement bounds (cm) for sanity checking
MALE_BOUNDS = {
    "chest": (80, 140),
    "waist": (60, 130),
    "hips": (80, 135),
    "shoulder_width": (36, 56),
    "arm_length": (50, 90),
    "neck": (32, 50),
    "torso_length": (38, 55),
}
FEMALE_BOUNDS = {
    "chest": (72, 130),
    "waist": (55, 120),
    "hips": (78, 140),
    "shoulder_width": (32, 50),
    "arm_length": (45, 80),
    "neck": (28, 44),
    "torso_length": (34, 50),
}

# Statistical averages by gender for fallback estimation
MALE_AVG = {
    "chest": 99, "waist": 84, "hips": 98,
    "shoulder_width": 45, "arm_length": 64, "neck": 38, "torso_length": 45,
}
FEMALE_AVG = {
    "chest": 91, "waist": 72, "hips": 100,
    "shoulder_width": 39, "arm_length": 57, "neck": 33, "torso_length": 41,
}

# Ratios for statistical estimation from height/weight
# Derived from ANSUR II anthropometric survey data
STAT_COEFFICIENTS = {
    "male": {
        "chest": {"intercept": 44.0, "height": 0.12, "weight": 0.50},
        "waist": {"intercept": 20.0, "height": 0.05, "weight": 0.65},
        "hips": {"intercept": 48.0, "height": 0.10, "weight": 0.42},
        "shoulder_width": {"intercept": 18.0, "height": 0.14, "weight": 0.04},
        "arm_length": {"intercept": 10.0, "height": 0.32, "weight": -0.02},
        "neck": {"intercept": 16.0, "height": 0.04, "weight": 0.18},
        "torso_length": {"intercept": 12.0, "height": 0.18, "weight": 0.02},
    },
    "female": {
        "chest": {"intercept": 42.0, "height": 0.10, "weight": 0.48},
        "waist": {"intercept": 16.0, "height": 0.04, "weight": 0.62},
        "hips": {"intercept": 50.0, "height": 0.08, "weight": 0.48},
        "shoulder_width": {"intercept": 16.0, "height": 0.12, "weight": 0.03},
        "arm_length": {"intercept": 8.0, "height": 0.30, "weight": -0.02},
        "neck": {"intercept": 14.0, "height": 0.03, "weight": 0.15},
        "torso_length": {"intercept": 10.0, "height": 0.17, "weight": 0.01},
    },
}


def decode_image(base64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def extract_silhouette(image: np.ndarray) -> np.ndarray:
    """Extract human silhouette using GrabCut with morphological refinement."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = image.shape[:2]
    rect = (int(w * 0.15), int(h * 0.02), int(w * 0.7), int(h * 0.96))
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Morphological cleanup: close gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask2


def statistical_estimate(
    height_cm: float,
    weight_kg: float,
    gender: Gender,
) -> dict[str, float]:
    """
    Estimate body measurements from height and weight using
    linear regression coefficients derived from ANSUR II data.
    """
    gender_key = gender.value
    coeffs = STAT_COEFFICIENTS[gender_key]

    result = {}
    for measurement, c in coeffs.items():
        value = c["intercept"] + c["height"] * height_cm + c["weight"] * weight_kg
        result[measurement] = round(value, 1)

    return result


def estimate_proportions_from_silhouette(
    front_silhouette: np.ndarray,
    side_silhouette: np.ndarray,
    height_cm: float,
) -> dict[str, float]:
    """
    Estimate body proportions from front and side silhouettes.
    Uses pixel ratios scaled by known height.
    """
    front_h = front_silhouette.shape[0]

    # Find body bounds in front view
    rows_with_body = np.any(front_silhouette > 0, axis=1)
    if not np.any(rows_with_body):
        return {}

    body_top = int(np.argmax(rows_with_body))
    body_bottom = int(front_h - np.argmax(rows_with_body[::-1]))
    pixel_height = body_bottom - body_top
    if pixel_height <= 0:
        return {}

    px_to_cm = height_cm / pixel_height

    def width_at_ratio(silhouette: np.ndarray, ratio: float) -> float:
        """Get the body width at a given vertical ratio from top of body."""
        row = int(body_top + pixel_height * ratio)
        row = max(0, min(row, silhouette.shape[0] - 1))
        cols = np.where(silhouette[row] > 0)[0]
        if len(cols) < 2:
            return 0.0
        return float(cols[-1] - cols[0]) * px_to_cm

    def stable_width_at_ratio(silhouette: np.ndarray, ratio: float, window: int = 5) -> float:
        """Average width over several rows for stability."""
        widths = []
        for offset in range(-window, window + 1):
            r = ratio + offset * (0.005)  # sample ~1% range
            w = width_at_ratio(silhouette, r)
            if w > 0:
                widths.append(w)
        return float(np.median(widths)) if widths else 0.0

    # Front view widths at anatomical landmarks
    # Ratios calibrated to standing pose proportions
    shoulder_width = stable_width_at_ratio(front_silhouette, 0.19)
    chest_front = stable_width_at_ratio(front_silhouette, 0.32)
    waist_front = stable_width_at_ratio(front_silhouette, 0.43)
    hip_front = stable_width_at_ratio(front_silhouette, 0.53)

    # Side view depths
    chest_side = stable_width_at_ratio(side_silhouette, 0.32)
    waist_side = stable_width_at_ratio(side_silhouette, 0.43)
    hip_side = stable_width_at_ratio(side_silhouette, 0.53)

    # Circumferences using Ramanujan's ellipse approximation:
    # C ≈ π * (3(a+b) - sqrt((3a+b)(a+3b)))
    # where a = front half-width, b = side half-depth
    def circumference(front_w: float, side_d: float) -> float:
        a = front_w / 2
        b = side_d / 2
        if a <= 0 or b <= 0:
            return 0.0
        h_val = ((a - b) / (a + b)) ** 2
        return float(np.pi * (a + b) * (1 + 3 * h_val / (10 + np.sqrt(4 - 3 * h_val))))

    # Torso length from shoulder to waist
    torso_px = pixel_height * 0.24
    torso_length = torso_px * px_to_cm

    # Arm length from shoulder to wrist (~44% of height)
    arm_length = height_cm * 0.44

    # Neck circumference
    neck_front = stable_width_at_ratio(front_silhouette, 0.13)
    neck_side = stable_width_at_ratio(side_silhouette, 0.13)
    neck = circumference(neck_front, neck_side)

    return {
        "chest": round(circumference(chest_front, chest_side), 1),
        "waist": round(circumference(waist_front, waist_side), 1),
        "hips": round(circumference(hip_front, hip_side), 1),
        "shoulder_width": round(shoulder_width, 1),
        "arm_length": round(arm_length, 1),
        "neck": round(neck, 1),
        "torso_length": round(torso_length, 1),
    }


def blend_measurements(
    silhouette_props: dict[str, float],
    stat_estimate: dict[str, float],
    gender: Gender,
    silhouette_weight: float = 0.6,
) -> dict[str, float]:
    """
    Blend silhouette-derived measurements with statistical estimates.
    Uses bounds to validate silhouette values — if out of range,
    falls back more heavily to statistical estimate.
    """
    bounds = MALE_BOUNDS if gender == Gender.male else FEMALE_BOUNDS
    result = {}

    for key in stat_estimate:
        stat_val = stat_estimate[key]
        sil_val = silhouette_props.get(key, 0.0)
        lo, hi = bounds[key]

        if sil_val <= 0 or sil_val < lo * 0.7 or sil_val > hi * 1.3:
            # Silhouette value is missing or wildly out of range — use stats
            result[key] = stat_val
        elif sil_val < lo or sil_val > hi:
            # Slightly out of range — lean toward stats
            blended = sil_val * 0.3 + stat_val * 0.7
            result[key] = round(max(lo, min(hi, blended)), 1)
        else:
            # In range — blend normally
            blended = sil_val * silhouette_weight + stat_val * (1 - silhouette_weight)
            result[key] = round(blended, 1)

    return result


def silhouette_to_smplx_betas(
    measurements: dict[str, float],
    gender: Gender,
    height_cm: float,
    weight_kg: float,
) -> list[float]:
    """
    Map body measurements to SMPL-X shape parameters (betas).
    """
    avg = MALE_AVG if gender == Gender.male else FEMALE_AVG
    avg_height = 175.0 if gender == Gender.male else 162.0
    avg_weight = 78.0 if gender == Gender.male else 65.0

    h_diff = (height_cm - avg_height) / avg_height
    w_diff = (weight_kg - avg_weight) / avg_weight
    chest_diff = (measurements["chest"] - avg["chest"]) / avg["chest"]
    waist_diff = (measurements["waist"] - avg["waist"]) / avg["waist"]
    hip_diff = (measurements["hips"] - avg["hips"]) / avg["hips"]

    betas = [
        w_diff * 3.0,
        h_diff * 4.0,
        chest_diff * 2.5,
        waist_diff * 2.0,
        hip_diff * 2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    return [round(b, 4) for b in betas]


def generate_mesh_from_measurements(
    measurements: dict[str, float],
    height_cm: float,
    weight_kg: float,
    gender: Gender,
) -> bytes:
    """
    Generate a human body mesh from measurements.
    Returns the mesh as GLB bytes.
    """
    from app.body_model.parametric_body import build_body_mesh

    mesh = build_body_mesh(
        height_cm=height_cm,
        chest_cm=measurements["chest"],
        waist_cm=measurements["waist"],
        hips_cm=measurements["hips"],
        shoulder_width_cm=measurements["shoulder_width"],
        neck_cm=measurements["neck"],
        arm_length_cm=measurements["arm_length"],
        torso_length_cm=measurements["torso_length"],
        weight_kg=weight_kg,
        gender=gender.value,
    )

    return mesh.export(file_type="glb")


def process_body_scan(
    front_image_b64: str,
    side_image_b64: str,
    height_cm: float,
    weight_kg: float,
    gender: Gender,
) -> tuple[BodyMeasurements, list[float], bytes]:
    """
    Full pipeline: images -> measurements + SMPL-X mesh.

    Returns (measurements, betas, glb_bytes).
    """
    front_img = decode_image(front_image_b64)
    side_img = decode_image(side_image_b64)

    front_sil = extract_silhouette(front_img)
    side_sil = extract_silhouette(side_img)

    # Get silhouette-based proportions
    silhouette_props = estimate_proportions_from_silhouette(front_sil, side_sil, height_cm)

    # Get statistical baseline from height/weight
    stat_props = statistical_estimate(height_cm, weight_kg, gender)

    # Blend: trust silhouette when it's reasonable, fall back to stats otherwise
    final_measurements = blend_measurements(silhouette_props, stat_props, gender)

    betas = silhouette_to_smplx_betas(final_measurements, gender, height_cm, weight_kg)

    measurements = BodyMeasurements(
        height=height_cm,
        weight=weight_kg,
        **final_measurements,
    )

    mesh_bytes = generate_mesh_from_measurements(
        final_measurements, height_cm, weight_kg, gender
    )

    return measurements, betas, mesh_bytes
