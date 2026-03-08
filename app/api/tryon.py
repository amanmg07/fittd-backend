from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import TryOnRequest, TryOnResult, SizeRecommendation
from app.services.fit_engine import recommend_size
from app.garment.mesh_generator import build_tryon_scene
from app.api.body import _profiles, _mesh_store, _photo_store
from app.api.garments import _garment_cache

router = APIRouter()

_scene_store: dict[str, bytes] = {}


def _pick_best_garment_image(image_urls: list[str]) -> str:
    """
    Pick the best garment image for VTON.
    Prefer high-res product-only flat-lay images over model-on shots.
    Nike flat-lay images tend to have specific URL patterns.
    """
    if not image_urls:
        raise ValueError("No garment images available")

    # Priority tiers for Nike CDN images
    # 1. High-res product images (flat-lay, no model)
    high_res_keys = ["t_PDP_1728", "t_PDP_1280", "t_default"]
    # 2. Medium-res product images
    mid_res_keys = ["t_web_pdp_936", "t_web_pdp_535", "t_PDP"]
    # 3. Any Nike CDN image with product indicators

    for keys in [high_res_keys, mid_res_keys]:
        for url in image_urls:
            if any(k in url for k in keys):
                return url

    return image_urls[0]


def _build_garment_description(garment) -> str:
    """
    Build a detailed description for the VTON model to preserve
    the exact look of the garment (color, style, logos, material).
    """
    parts = [garment.brand, garment.name]

    if garment.color:
        parts.append(garment.color)

    parts.append(f"{garment.fit_type} fit")

    # Add material info
    materials = []
    for mat, pct in garment.material_composition.items():
        materials.append(f"{int(pct * 100)}% {mat}")
    if materials:
        parts.append(", ".join(materials))

    return ", ".join(parts)


@router.post("/", response_model=TryOnResult)
async def virtual_tryon(request: TryOnRequest):
    """
    Perform virtual try-on: recommend size, generate 3D scene
    with garment draped on user's body model.
    """
    profile = _profiles.get(request.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Body profile not found. Scan first.")

    garment = _garment_cache.get(request.product_id)
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found. Scrape it first.")

    # Get size recommendation
    recommendation = recommend_size(profile.measurements, garment)
    selected_size = request.size or recommendation.recommended_size

    # Find the size dimensions
    size_dims = None
    for gs in garment.sizes:
        if gs.size_label == selected_size:
            size_dims = gs.model_dump()
            break

    if not size_dims:
        raise HTTPException(status_code=400, detail=f"Size '{selected_size}' not available")

    # Build the 3D try-on scene
    body_mesh_bytes = _mesh_store.get(request.user_id)
    if not body_mesh_bytes:
        raise HTTPException(status_code=404, detail="Body mesh not found")

    # Pick best garment image for 3D texture
    garment_image_url = None
    if garment.image_urls:
        garment_image_url = _pick_best_garment_image(garment.image_urls)

    try:
        scene_bytes = build_tryon_scene(
            body_glb=body_mesh_bytes,
            garment_size_dims=size_dims,
            garment_color="#333333",
            garment_image_url=garment_image_url,
        )

        scene_key = f"{request.user_id}_{request.product_id}_{selected_size}"
        _scene_store[scene_key] = scene_bytes

        return TryOnResult(
            user_id=request.user_id,
            product_id=request.product_id,
            selected_size=selected_size,
            recommendation=recommendation,
            scene_url=f"/api/tryon/scene/{scene_key}.glb",
            fit_map_url=f"/api/tryon/scene/{scene_key}.glb",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene generation failed: {str(e)}")


@router.get("/scene/{scene_key}.glb")
async def get_tryon_scene(scene_key: str):
    """Serve the 3D try-on scene."""
    scene_bytes = _scene_store.get(scene_key)
    if not scene_bytes:
        raise HTTPException(status_code=404, detail="Scene not found")
    return Response(content=scene_bytes, media_type="model/gltf-binary")


@router.post("/ai")
async def ai_virtual_tryon(request: TryOnRequest):
    """
    AI-powered virtual try-on using IDM-VTON.
    Returns a realistic composite image of the user wearing the garment.
    """
    from app.services.vton import try_on_image_async

    # Prefer direct photo from capture, fall back to stored scan photo
    person_photo = request.photo or _photo_store.get(request.user_id)
    if not person_photo:
        raise HTTPException(
            status_code=404,
            detail="No photo found. Please take a photo or scan your body first.",
        )

    garment = _garment_cache.get(request.product_id)
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found. Scrape it first.")

    if not garment.image_urls:
        raise HTTPException(status_code=400, detail="No garment image available.")

    profile = _profiles.get(request.user_id)
    recommendation = None
    if profile:
        recommendation = recommend_size(profile.measurements, garment)

    try:
        garment_url = _pick_best_garment_image(garment.image_urls)
        garment_desc = _build_garment_description(garment)

        result_b64 = await try_on_image_async(
            person_image_b64=person_photo,
            garment_image_url=garment_url,
            garment_description=garment_desc,
        )

        selected_size = request.size or (
            recommendation.recommended_size if recommendation else garment.sizes[0].size_label
        )

        return {
            "image_b64": result_b64,
            "selected_size": selected_size,
            "recommendation": recommendation.model_dump() if recommendation else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI try-on failed: {str(e)}")
