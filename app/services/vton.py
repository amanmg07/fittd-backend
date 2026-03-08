"""
AI Virtual Try-On using IDM-VTON via Replicate.

Sends a person photo + garment image to the model and gets back
a realistic composite image of the person wearing the garment.

Uses MediaPipe Pose for body-aware cropping to ensure the person
is properly framed for the VTON model.
"""

import base64
import tempfile
import os
from PIL import Image, ImageOps
import io
import httpx
import cv2
import numpy as np
import mediapipe as mp
import replicate


# Keep gradio_client for background removal (free HF Space)
from gradio_client import Client, handle_file

_rembg_client: Client | None = None


def get_rembg_client() -> Client:
    global _rembg_client
    if _rembg_client is None:
        _rembg_client = Client("not-lain/background-removal")
    return _rembg_client


def remove_background(image_path: str) -> str:
    """
    Remove background using a free HF Space.
    Returns path to the result image (PNG with transparency).
    """
    try:
        client = get_rembg_client()
        result = client.predict(
            handle_file(image_path),
            api_name="/predict",
        )
        return result
    except Exception:
        return image_path


def detect_body_bounds(image_cv: np.ndarray) -> dict | None:
    """
    Use MediaPipe Pose to detect body landmarks and return bounding info.
    Returns dict with top_y, bottom_y, center_x, shoulder_width or None.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = results.pose_landmarks.landmark
        h, w = image_cv.shape[:2]

        # Collect all visible landmark positions
        visible_xs = []
        visible_ys = []
        for lm in landmarks:
            if lm.visibility >= 0.5:
                visible_xs.append(lm.x * w)
                visible_ys.append(lm.y * h)

        if len(visible_xs) < 5:
            return None

        # Key landmarks for framing
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Center X from mid-shoulder
        center_x = (left_shoulder.x + right_shoulder.x) / 2 * w

        # Top: above the head (estimate from nose)
        head_top_y = nose.y * h
        if landmarks[7].visibility >= 0.5:  # left ear
            ear_to_nose = abs(nose.y - landmarks[7].y) * h
            head_top_y = nose.y * h - ear_to_nose * 1.5
        else:
            head_top_y = nose.y * h - (h * 0.04)

        # Bottom: ankles or lowest visible point
        bottom_y = max(visible_ys)
        if left_ankle.visibility >= 0.5 and right_ankle.visibility >= 0.5:
            bottom_y = max(left_ankle.y, right_ankle.y) * h

        # Shoulder width in pixels
        shoulder_w = abs(left_shoulder.x - right_shoulder.x) * w

        # Hip width in pixels
        hip_w = abs(left_hip.x - right_hip.x) * w

        return {
            "top_y": head_top_y,
            "bottom_y": bottom_y,
            "center_x": center_x,
            "shoulder_width": shoulder_w,
            "hip_width": hip_w,
            "body_width": max(shoulder_w, hip_w),
        }


def body_aware_crop(img: Image.Image, target_w: int = 768, target_h: int = 1024) -> Image.Image:
    """
    Crop and resize person image using pose detection for proper framing.
    Ensures the person is centered and fully visible in the output.
    """
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bounds = detect_body_bounds(img_cv)

    w, h = img.size

    if bounds is None:
        # Fallback: basic center crop
        target_aspect = target_w / target_h
        aspect = w / h
        if aspect > target_aspect:
            new_w = int(h * target_aspect)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_aspect)
            img = img.crop((0, 0, w, min(new_h, h)))
        return img.resize((target_w, target_h), Image.LANCZOS)

    # Calculate crop region centered on the body
    body_height = bounds["bottom_y"] - bounds["top_y"]
    body_width = bounds["body_width"]
    center_x = bounds["center_x"]

    # Add padding: 15% above head, 5% below feet, 40% on each side of body
    pad_top = body_height * 0.15
    pad_bottom = body_height * 0.05
    pad_side = body_width * 0.40

    crop_top = bounds["top_y"] - pad_top
    crop_bottom = bounds["bottom_y"] + pad_bottom
    crop_height = crop_bottom - crop_top

    # Width from target aspect ratio
    target_aspect = target_w / target_h
    crop_width = crop_height * target_aspect

    # Ensure crop is wide enough to include the body + padding
    min_crop_width = body_width + 2 * pad_side
    if crop_width < min_crop_width:
        crop_width = min_crop_width
        crop_height = crop_width / target_aspect
        # Re-center vertically
        body_center_y = (bounds["top_y"] + bounds["bottom_y"]) / 2
        crop_top = body_center_y - crop_height * 0.45  # slightly above center
        crop_bottom = crop_top + crop_height

    crop_left = center_x - crop_width / 2
    crop_right = center_x + crop_width / 2

    # Clamp to image bounds
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_right > w:
        crop_left -= (crop_right - w)
        crop_right = w
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_bottom > h:
        crop_top -= (crop_bottom - h)
        crop_bottom = h

    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(w, crop_right)
    crop_bottom = min(h, crop_bottom)

    img = img.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
    return img.resize((target_w, target_h), Image.LANCZOS)


def preprocess_person_image(image_b64: str) -> str:
    """
    Preprocess person image for IDM-VTON:
    1. Apply EXIF rotation
    2. Remove background via HF Space
    3. Paste onto neutral gray background
    4. Body-aware crop and resize to 768x1024 portrait
    """
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes))

    # Apply EXIF orientation
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Save temp for bg removal
    tmp_in = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp_in.name, "JPEG", quality=90)

    # Remove background
    bg_removed_path = remove_background(tmp_in.name)

    try:
        bg_img = Image.open(bg_removed_path)
        if bg_img.mode == "RGBA":
            # Paste onto neutral gray background
            bg = Image.new("RGB", bg_img.size, (220, 220, 220))
            bg.paste(bg_img, mask=bg_img.split()[3])
            img = bg
        else:
            img = bg_img.convert("RGB")
    except Exception:
        pass  # Use original image if bg removal result can't be loaded

    os.unlink(tmp_in.name)

    # Body-aware crop to 768x1024
    img = body_aware_crop(img, 768, 1024)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=92)
    return tmp.name


def preprocess_garment_image(image_url: str) -> str:
    """
    Download and preprocess garment image for IDM-VTON:
    1. Download from URL (high quality)
    2. Remove background via HF Space
    3. Auto-crop to garment bounding box
    4. Resize to 768x1024 with white padding, garment centered and large
    """
    with httpx.Client(timeout=30.0, follow_redirects=True) as http:
        resp = http.get(image_url)
        resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content))
    img = ImageOps.exif_transpose(img)

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Save temp for bg removal
    tmp_in = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp_in.name, "JPEG", quality=95)

    # Remove background
    bg_removed_path = remove_background(tmp_in.name)

    try:
        bg_img = Image.open(bg_removed_path)
        if bg_img.mode == "RGBA":
            # Auto-crop to garment bounding box (trim transparent edges)
            bbox = bg_img.split()[3].getbbox()
            if bbox:
                bg_img = bg_img.crop(bbox)
            bg = Image.new("RGB", bg_img.size, (255, 255, 255))
            bg.paste(bg_img, mask=bg_img.split()[3])
            img = bg
        else:
            img = bg_img.convert("RGB")
    except Exception:
        pass

    os.unlink(tmp_in.name)

    # Resize to 768x1024 with white padding
    # Make garment fill ~85% of the frame for better detail preservation
    target_w, target_h = 768, 1024
    fill_w = int(target_w * 0.85)
    fill_h = int(target_h * 0.85)

    # Scale garment to fill the target area while keeping aspect ratio
    scale = min(fill_w / img.width, fill_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded.paste(img, (x_offset, y_offset))

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    padded.save(tmp.name, "JPEG", quality=95)
    return tmp.name


def _image_path_to_data_uri(path: str) -> str:
    """Convert a local image file to a data URI for Replicate input."""
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def try_on_image(
    person_image_b64: str,
    garment_image_url: str,
    garment_description: str = "",
    denoise_steps: int = 20,
    seed: int = 42,
) -> str:
    """
    Run AI virtual try-on via Replicate (cuuupid/idm-vton).
    Returns base64 encoded result image.
    """
    person_path = preprocess_person_image(person_image_b64)
    garment_path = preprocess_garment_image(garment_image_url)

    try:
        person_uri = _image_path_to_data_uri(person_path)
        garment_uri = _image_path_to_data_uri(garment_path)

        output = replicate.run(
            "cuuupid/idm-vton",
            input={
                "human_img": person_uri,
                "garm_img": garment_uri,
                "garment_des": garment_description,
                "category": "upper_body",
                "crop": False,
                "seed": seed,
                "steps": denoise_steps,
            },
        )

        # Replicate returns a URL to the output image
        result_url = output if isinstance(output, str) else str(output)

        # Download the result image
        with httpx.Client(timeout=60.0, follow_redirects=True) as http:
            resp = http.get(result_url)
            resp.raise_for_status()

        result_b64 = base64.b64encode(resp.content).decode("utf-8")
        return result_b64

    finally:
        os.unlink(person_path)
        os.unlink(garment_path)


def mirror_image_b64(image_b64: str) -> str:
    """Horizontally mirror an image. Used to approximate a back view from front."""
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes))
    mirrored = ImageOps.mirror(img)
    buf = io.BytesIO()
    mirrored.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def try_on_image_async(
    person_image_b64: str,
    garment_image_url: str,
    garment_description: str = "",
) -> str:
    """
    Async wrapper — runs the blocking Replicate call in a thread.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        try_on_image,
        person_image_b64,
        garment_image_url,
        garment_description,
    )


# --- Novel View Synthesis (360° rotation) ---


def generate_novel_views(front_image_b64: str, num_views: int = 12) -> list[dict]:
    """
    Generate multiple rotated views from a single front image using
    Zero123++ on Replicate.
    Returns a list of {"angle_deg": int, "image_b64": str}.
    """
    img_bytes = base64.b64decode(front_image_b64)

    # Build data URI for Replicate
    input_uri = f"data:image/png;base64,{front_image_b64}"

    try:
        output = replicate.run(
            "jd7h/zero123plus",
            input={
                "image": input_uri,
                "num_inference_steps": 75,
                "guidance_scale": 10.0,
            },
        )

        result_url = output if isinstance(output, str) else str(output)

        # Download the result grid
        with httpx.Client(timeout=60.0, follow_redirects=True) as http:
            resp = http.get(result_url)
            resp.raise_for_status()

        result_img = Image.open(io.BytesIO(resp.content))

        # Zero123++ outputs a 3x2 grid (3 cols x 2 rows)
        grid_w, grid_h = result_img.size
        cell_w = grid_w // 3
        cell_h = grid_h // 2

        # The 6 views are approximately:
        # Row 0: 30°, 90°, 150°
        # Row 1: 210°, 270°, 330°
        angle_map = [
            (0, 0, 30), (1, 0, 90), (2, 0, 150),
            (0, 1, 210), (1, 1, 270), (2, 1, 330),
        ]

        views = []
        # Include the original front as 0°
        img = Image.open(io.BytesIO(img_bytes))
        buf = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="PNG")
        views.append({"angle_deg": 0, "image_b64": base64.b64encode(buf.getvalue()).decode()})

        for col, row, angle in angle_map:
            cell = result_img.crop((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h))
            if cell.mode != "RGB":
                cell = cell.convert("RGB")
            buf = io.BytesIO()
            cell.save(buf, format="PNG")
            views.append({
                "angle_deg": angle,
                "image_b64": base64.b64encode(buf.getvalue()).decode(),
            })

        # Sort by angle
        views.sort(key=lambda v: v["angle_deg"])

        # Add 180° by mirroring 0° if not present
        filled = {v["angle_deg"]: v for v in views}
        if 180 not in filled:
            mirrored = mirror_image_b64(front_image_b64)
            filled[180] = {"angle_deg": 180, "image_b64": mirrored}

        final_views = sorted(filled.values(), key=lambda v: v["angle_deg"])
        return final_views

    except Exception:
        # Fallback: generate basic views from the front image using mirrors
        views = [
            {"angle_deg": 0, "image_b64": front_image_b64},
            {"angle_deg": 180, "image_b64": mirror_image_b64(front_image_b64)},
        ]
        return views


async def generate_novel_views_async(front_image_b64: str, num_views: int = 12) -> list[dict]:
    """Async wrapper for novel view generation."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        generate_novel_views,
        front_image_b64,
        num_views,
    )
