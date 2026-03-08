"""
Body scan processor: takes front + side photos and produces
a SMPL-X body model with measurements.

Pipeline:
1. Detect body landmarks using MediaPipe Pose (33 keypoints)
2. Calculate measurements from landmark positions + side silhouette depth
3. Blend with statistical estimates for robustness
4. Generate SMPL-X mesh with derived shape parameters
"""

import base64
import io
import numpy as np
import cv2
import mediapipe as mp

from app.models.schemas import BodyMeasurements, Gender


# MediaPipe Pose landmark indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8

# Minimum visibility score to trust a landmark
MIN_VISIBILITY = 0.5

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

# Typical depth-to-width ratios by body region (front width -> side depth)
# Based on anthropometric data; used when side landmarks aren't available
DEPTH_RATIOS = {
    "chest": {"male": 0.75, "female": 0.70},
    "waist": {"male": 0.80, "female": 0.72},
    "hips": {"male": 0.70, "female": 0.75},
    "neck": {"male": 0.85, "female": 0.82},
}


def decode_image(base64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def detect_pose_landmarks(image: np.ndarray) -> list | None:
    """
    Detect body pose landmarks using MediaPipe Pose.
    Returns list of 33 landmarks with x, y, z, visibility, or None if detection fails.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        return results.pose_landmarks.landmark


def landmark_pixel(landmark, img_h: int, img_w: int) -> tuple[float, float]:
    """Convert normalized landmark coordinates to pixel coordinates."""
    return landmark.x * img_w, landmark.y * img_h


def landmark_distance_px(lm1, lm2, img_h: int, img_w: int) -> float:
    """Euclidean distance between two landmarks in pixels."""
    x1, y1 = landmark_pixel(lm1, img_h, img_w)
    x2, y2 = landmark_pixel(lm2, img_h, img_w)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def landmarks_visible(landmarks: list, indices: list[int]) -> bool:
    """Check if all specified landmarks have sufficient visibility."""
    return all(landmarks[i].visibility >= MIN_VISIBILITY for i in indices)


def extract_silhouette(image: np.ndarray) -> np.ndarray:
    """Extract human silhouette using GrabCut with morphological refinement."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = image.shape[:2]
    rect = (int(w * 0.15), int(h * 0.02), int(w * 0.7), int(h * 0.96))
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask2


def width_at_row(silhouette: np.ndarray, row: int, px_to_cm: float) -> float:
    """Get body width in cm at a specific pixel row from a silhouette."""
    row = max(0, min(row, silhouette.shape[0] - 1))
    # Average over a small window for stability
    rows_to_check = range(max(0, row - 3), min(silhouette.shape[0], row + 4))
    widths = []
    for r in rows_to_check:
        cols = np.where(silhouette[r] > 0)[0]
        if len(cols) >= 2:
            widths.append(float(cols[-1] - cols[0]) * px_to_cm)
    return float(np.median(widths)) if widths else 0.0


def circumference_from_widths(front_width_cm: float, side_depth_cm: float) -> float:
    """
    Estimate circumference from front width and side depth using
    Ramanujan's ellipse perimeter approximation.
    """
    a = front_width_cm / 2
    b = side_depth_cm / 2
    if a <= 0 or b <= 0:
        return 0.0
    h_val = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1 + 3 * h_val / (10 + np.sqrt(4 - 3 * h_val))))


def estimate_from_landmarks(
    front_landmarks: list,
    front_image: np.ndarray,
    side_image: np.ndarray,
    height_cm: float,
    gender: Gender,
) -> dict[str, float]:
    """
    Estimate body measurements using MediaPipe Pose landmarks from the front image
    combined with silhouette width from the side image for depth.
    """
    img_h, img_w = front_image.shape[:2]
    gender_key = gender.value

    # Calculate pixel-to-cm ratio from known height
    # Use ankle-to-top-of-head distance for calibration
    if landmarks_visible(front_landmarks, [LM_NOSE, LM_LEFT_ANKLE, LM_RIGHT_ANKLE]):
        # Top of head is roughly 10% of head height above nose
        _, nose_y = landmark_pixel(front_landmarks[LM_NOSE], img_h, img_w)
        _, left_ankle_y = landmark_pixel(front_landmarks[LM_LEFT_ANKLE], img_h, img_w)
        _, right_ankle_y = landmark_pixel(front_landmarks[LM_RIGHT_ANKLE], img_h, img_w)
        ankle_y = (left_ankle_y + right_ankle_y) / 2

        # Estimate head top from ear-to-nose distance
        if landmarks_visible(front_landmarks, [LM_LEFT_EAR]):
            _, ear_y = landmark_pixel(front_landmarks[LM_LEFT_EAR], img_h, img_w)
            head_above_nose = abs(nose_y - ear_y) * 1.2
        else:
            head_above_nose = (ankle_y - nose_y) * 0.06

        head_top_y = nose_y - head_above_nose
        pixel_height = ankle_y - head_top_y
    else:
        # Fallback: use image bounds
        pixel_height = img_h * 0.9

    if pixel_height <= 0:
        return {}

    px_to_cm = height_cm / pixel_height

    # --- Shoulder width ---
    shoulder_width = 0.0
    if landmarks_visible(front_landmarks, [LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER]):
        shoulder_width = landmark_distance_px(
            front_landmarks[LM_LEFT_SHOULDER],
            front_landmarks[LM_RIGHT_SHOULDER],
            img_h, img_w,
        ) * px_to_cm

    # --- Arm length (shoulder -> elbow -> wrist) ---
    arm_length = 0.0
    # Try left arm first, then right
    for shoulder_idx, elbow_idx, wrist_idx in [
        (LM_LEFT_SHOULDER, LM_LEFT_ELBOW, LM_LEFT_WRIST),
        (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
    ]:
        if landmarks_visible(front_landmarks, [shoulder_idx, elbow_idx, wrist_idx]):
            upper = landmark_distance_px(
                front_landmarks[shoulder_idx], front_landmarks[elbow_idx], img_h, img_w
            )
            lower = landmark_distance_px(
                front_landmarks[elbow_idx], front_landmarks[wrist_idx], img_h, img_w
            )
            arm_length = (upper + lower) * px_to_cm
            break

    # --- Torso length (mid-shoulder to mid-hip) ---
    torso_length = 0.0
    if landmarks_visible(front_landmarks, [
        LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP,
    ]):
        _, ls_y = landmark_pixel(front_landmarks[LM_LEFT_SHOULDER], img_h, img_w)
        _, rs_y = landmark_pixel(front_landmarks[LM_RIGHT_SHOULDER], img_h, img_w)
        _, lh_y = landmark_pixel(front_landmarks[LM_LEFT_HIP], img_h, img_w)
        _, rh_y = landmark_pixel(front_landmarks[LM_RIGHT_HIP], img_h, img_w)
        mid_shoulder_y = (ls_y + rs_y) / 2
        mid_hip_y = (lh_y + rh_y) / 2
        torso_length = abs(mid_hip_y - mid_shoulder_y) * px_to_cm

    # --- Circumferences: use landmark Y positions to sample silhouette widths ---
    # Extract side silhouette for depth measurements
    side_sil = extract_silhouette(side_image)
    side_h, side_w = side_image.shape[:2]
    side_px_to_cm = height_cm / (side_h * 0.9)  # approximate

    # If we have side landmarks, use them for better calibration
    side_landmarks = detect_pose_landmarks(side_image)
    if side_landmarks and landmarks_visible(side_landmarks, [LM_NOSE, LM_LEFT_ANKLE]):
        _, s_nose_y = landmark_pixel(side_landmarks[LM_NOSE], side_h, side_w)
        _, s_ankle_y = landmark_pixel(side_landmarks[LM_LEFT_ANKLE], side_h, side_w)
        s_pixel_height = s_ankle_y - s_nose_y
        if s_pixel_height > 0:
            # Nose to ankle is roughly 90% of height
            side_px_to_cm = (height_cm * 0.90) / s_pixel_height

    # Also get front silhouette for front widths
    front_sil = extract_silhouette(front_image)

    # Chest: at the level ~40% between shoulder and hip
    chest_cm = 0.0
    if landmarks_visible(front_landmarks, [
        LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP,
    ]):
        _, ls_y = landmark_pixel(front_landmarks[LM_LEFT_SHOULDER], img_h, img_w)
        _, rs_y = landmark_pixel(front_landmarks[LM_RIGHT_SHOULDER], img_h, img_w)
        _, lh_y = landmark_pixel(front_landmarks[LM_LEFT_HIP], img_h, img_w)
        _, rh_y = landmark_pixel(front_landmarks[LM_RIGHT_HIP], img_h, img_w)
        mid_shoulder_y = (ls_y + rs_y) / 2
        mid_hip_y = (lh_y + rh_y) / 2

        # Chest at 30% down from shoulders to hips
        chest_row_front = int(mid_shoulder_y + (mid_hip_y - mid_shoulder_y) * 0.30)
        chest_front_w = width_at_row(front_sil, chest_row_front, px_to_cm)

        # Map front row ratio to side image
        front_ratio = chest_row_front / img_h
        chest_row_side = int(front_ratio * side_h)
        chest_side_d = width_at_row(side_sil, chest_row_side, side_px_to_cm)

        if chest_side_d <= 0:
            chest_side_d = chest_front_w * DEPTH_RATIOS["chest"][gender_key]

        chest_cm = circumference_from_widths(chest_front_w, chest_side_d)

    # Waist: at ~65% between shoulder and hip
    waist_cm = 0.0
    if landmarks_visible(front_landmarks, [
        LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP,
    ]):
        waist_row_front = int(mid_shoulder_y + (mid_hip_y - mid_shoulder_y) * 0.65)
        waist_front_w = width_at_row(front_sil, waist_row_front, px_to_cm)

        front_ratio = waist_row_front / img_h
        waist_row_side = int(front_ratio * side_h)
        waist_side_d = width_at_row(side_sil, waist_row_side, side_px_to_cm)

        if waist_side_d <= 0:
            waist_side_d = waist_front_w * DEPTH_RATIOS["waist"][gender_key]

        waist_cm = circumference_from_widths(waist_front_w, waist_side_d)

    # Hips: at ~110% of shoulder-to-hip distance (just below hip landmarks)
    hips_cm = 0.0
    if landmarks_visible(front_landmarks, [LM_LEFT_HIP, LM_RIGHT_HIP]):
        _, lh_y = landmark_pixel(front_landmarks[LM_LEFT_HIP], img_h, img_w)
        _, rh_y = landmark_pixel(front_landmarks[LM_RIGHT_HIP], img_h, img_w)
        hip_row_front = int((lh_y + rh_y) / 2)
        hip_front_w = width_at_row(front_sil, hip_row_front, px_to_cm)

        front_ratio = hip_row_front / img_h
        hip_row_side = int(front_ratio * side_h)
        hip_side_d = width_at_row(side_sil, hip_row_side, side_px_to_cm)

        if hip_side_d <= 0:
            hip_side_d = hip_front_w * DEPTH_RATIOS["hips"][gender_key]

        hips_cm = circumference_from_widths(hip_front_w, hip_side_d)

    # Neck: midpoint between ears/nose and shoulders
    neck_cm = 0.0
    if landmarks_visible(front_landmarks, [LM_NOSE, LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER]):
        _, nose_y = landmark_pixel(front_landmarks[LM_NOSE], img_h, img_w)
        _, ls_y = landmark_pixel(front_landmarks[LM_LEFT_SHOULDER], img_h, img_w)
        _, rs_y = landmark_pixel(front_landmarks[LM_RIGHT_SHOULDER], img_h, img_w)
        mid_shoulder_y = (ls_y + rs_y) / 2
        neck_row_front = int((nose_y + mid_shoulder_y) / 2)
        neck_front_w = width_at_row(front_sil, neck_row_front, px_to_cm)

        front_ratio = neck_row_front / img_h
        neck_row_side = int(front_ratio * side_h)
        neck_side_d = width_at_row(side_sil, neck_row_side, side_px_to_cm)

        if neck_side_d <= 0:
            neck_side_d = neck_front_w * DEPTH_RATIOS["neck"][gender_key]

        neck_cm = circumference_from_widths(neck_front_w, neck_side_d)

    result = {}
    if shoulder_width > 0:
        result["shoulder_width"] = round(shoulder_width, 1)
    if arm_length > 0:
        result["arm_length"] = round(arm_length, 1)
    if torso_length > 0:
        result["torso_length"] = round(torso_length, 1)
    if chest_cm > 0:
        result["chest"] = round(chest_cm, 1)
    if waist_cm > 0:
        result["waist"] = round(waist_cm, 1)
    if hips_cm > 0:
        result["hips"] = round(hips_cm, 1)
    if neck_cm > 0:
        result["neck"] = round(neck_cm, 1)

    return result


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


def blend_measurements(
    landmark_props: dict[str, float],
    stat_estimate: dict[str, float],
    gender: Gender,
    landmark_weight: float = 0.7,
) -> dict[str, float]:
    """
    Blend landmark-derived measurements with statistical estimates.
    Landmarks get higher trust (0.7) than the old silhouette approach (0.6)
    because MediaPipe provides actual joint positions.
    """
    bounds = MALE_BOUNDS if gender == Gender.male else FEMALE_BOUNDS
    result = {}

    for key in stat_estimate:
        stat_val = stat_estimate[key]
        lm_val = landmark_props.get(key, 0.0)
        lo, hi = bounds[key]

        if lm_val <= 0 or lm_val < lo * 0.7 or lm_val > hi * 1.3:
            # Landmark value is missing or wildly out of range — use stats
            result[key] = stat_val
        elif lm_val < lo or lm_val > hi:
            # Slightly out of range — lean toward stats
            blended = lm_val * 0.3 + stat_val * 0.7
            result[key] = round(max(lo, min(hi, blended)), 1)
        else:
            # In range — blend with higher trust for landmarks
            blended = lm_val * landmark_weight + stat_val * (1 - landmark_weight)
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

    # Try MediaPipe Pose landmark detection on front image
    front_landmarks = detect_pose_landmarks(front_img)

    if front_landmarks is not None:
        # Use landmark-based measurement extraction
        landmark_props = estimate_from_landmarks(
            front_landmarks, front_img, side_img, height_cm, gender,
        )
    else:
        # MediaPipe failed — fall back to silhouette-only approach
        landmark_props = _fallback_silhouette_estimate(front_img, side_img, height_cm)

    # Get statistical baseline from height/weight
    stat_props = statistical_estimate(height_cm, weight_kg, gender)

    # Blend: trust landmarks when reasonable, fall back to stats otherwise
    final_measurements = blend_measurements(landmark_props, stat_props, gender)

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


def _fallback_silhouette_estimate(
    front_img: np.ndarray,
    side_img: np.ndarray,
    height_cm: float,
) -> dict[str, float]:
    """
    Fallback when MediaPipe can't detect a pose.
    Uses the old silhouette + hardcoded ratio approach.
    """
    front_sil = extract_silhouette(front_img)
    side_sil = extract_silhouette(side_img)

    front_h = front_sil.shape[0]
    rows_with_body = np.any(front_sil > 0, axis=1)
    if not np.any(rows_with_body):
        return {}

    body_top = int(np.argmax(rows_with_body))
    body_bottom = int(front_h - np.argmax(rows_with_body[::-1]))
    pixel_height = body_bottom - body_top
    if pixel_height <= 0:
        return {}

    px_to_cm = height_cm / pixel_height

    def stable_width(silhouette, ratio, window=5):
        widths = []
        for offset in range(-window, window + 1):
            row = int(body_top + pixel_height * (ratio + offset * 0.005))
            row = max(0, min(row, silhouette.shape[0] - 1))
            cols = np.where(silhouette[row] > 0)[0]
            if len(cols) >= 2:
                widths.append(float(cols[-1] - cols[0]) * px_to_cm)
        return float(np.median(widths)) if widths else 0.0

    def circ(front_w, side_d):
        a, b = front_w / 2, side_d / 2
        if a <= 0 or b <= 0:
            return 0.0
        h_val = ((a - b) / (a + b)) ** 2
        return float(np.pi * (a + b) * (1 + 3 * h_val / (10 + np.sqrt(4 - 3 * h_val))))

    return {
        "chest": round(circ(stable_width(front_sil, 0.32), stable_width(side_sil, 0.32)), 1),
        "waist": round(circ(stable_width(front_sil, 0.43), stable_width(side_sil, 0.43)), 1),
        "hips": round(circ(stable_width(front_sil, 0.53), stable_width(side_sil, 0.53)), 1),
        "shoulder_width": round(stable_width(front_sil, 0.19), 1),
        "arm_length": round(height_cm * 0.44, 1),
        "neck": round(circ(stable_width(front_sil, 0.13), stable_width(side_sil, 0.13)), 1),
        "torso_length": round(pixel_height * 0.24 * px_to_cm, 1),
    }
