"""
AI Virtual Try-On using IDM-VTON via Hugging Face Spaces.

Sends a person photo + garment image to the model and gets back
a realistic composite image of the person wearing the garment.
"""

import base64
import tempfile
import os
from PIL import Image, ImageOps
import io
import httpx
from gradio_client import Client, handle_file


_vton_client: Client | None = None
_rembg_client: Client | None = None


def get_vton_client() -> Client:
    global _vton_client
    if _vton_client is None:
        _vton_client = Client("yisol/IDM-VTON")
    return _vton_client


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
        # If bg removal fails, return original
        return image_path


def preprocess_person_image(image_b64: str) -> str:
    """
    Preprocess person image for IDM-VTON:
    1. Apply EXIF rotation
    2. Remove background via HF Space
    3. Paste onto neutral gray background
    4. Resize to 768x1024 portrait
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

    # Resize to 768x1024 portrait
    target_w, target_h = 768, 1024

    w, h = img.size
    aspect = w / h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        img = img.crop((0, 0, w, min(new_h, h)))

    img = img.resize((target_w, target_h), Image.LANCZOS)

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


def try_on_image(
    person_image_b64: str,
    garment_image_url: str,
    garment_description: str = "",
    denoise_steps: int = 20,
    seed: int = 42,
) -> str:
    """
    Run AI virtual try-on.
    Returns base64 encoded result image.
    """
    person_path = preprocess_person_image(person_image_b64)
    garment_path = preprocess_garment_image(garment_image_url)

    try:
        client = get_vton_client()

        result = client.predict(
            dict={
                "background": handle_file(person_path),
                "layers": [],
                "composite": None,
            },
            garm_img=handle_file(garment_path),
            garment_des=garment_description,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=denoise_steps,
            seed=seed,
            api_name="/tryon",
        )

        result_path = result[0]

        with open(result_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode("utf-8")

        return result_b64

    finally:
        os.unlink(person_path)
        os.unlink(garment_path)


async def try_on_image_async(
    person_image_b64: str,
    garment_image_url: str,
    garment_description: str = "",
) -> str:
    """
    Async wrapper — runs the blocking gradio call in a thread.
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
