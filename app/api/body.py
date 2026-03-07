from fastapi import APIRouter, HTTPException
from app.models.schemas import BodyScanRequest, BodyProfile, BodyRestoreRequest
from app.body_model.scan_processor import process_body_scan

router = APIRouter()

# In-memory store for development (replace with DB in production)
_profiles: dict[str, BodyProfile] = {}


@router.post("/scan", response_model=BodyProfile)
async def create_body_scan(request: BodyScanRequest):
    """Process front + side photos to create a 3D body model."""
    try:
        measurements, betas, mesh_bytes = process_body_scan(
            front_image_b64=request.front_image,
            side_image_b64=request.side_image,
            height_cm=request.height_cm,
            weight_kg=request.weight_kg,
            gender=request.gender,
        )

        # In production: upload mesh_bytes to S3
        mesh_url = f"/api/body/{request.user_id}/mesh.glb"

        profile = BodyProfile(
            user_id=request.user_id,
            gender=request.gender,
            measurements=measurements,
            mesh_url=mesh_url,
            smplx_params={"betas": betas},
        )

        _profiles[request.user_id] = profile
        _mesh_store[request.user_id] = mesh_bytes
        _photo_store[request.user_id] = request.front_image

        return profile

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/restore", response_model=BodyProfile)
async def restore_body_profile(request: BodyRestoreRequest):
    """Restore a body profile from the client (e.g. after server restart)."""
    from app.body_model.scan_processor import generate_mesh_from_measurements

    profile = request.profile
    _profiles[profile.user_id] = profile

    if request.front_photo:
        _photo_store[profile.user_id] = request.front_photo

    m = profile.measurements
    mesh_bytes = generate_mesh_from_measurements(
        {
            "chest": m.chest,
            "waist": m.waist,
            "hips": m.hips,
            "shoulder_width": m.shoulder_width,
            "arm_length": m.arm_length,
            "neck": m.neck,
            "torso_length": m.torso_length,
        },
        height_cm=m.height,
        weight_kg=m.weight,
        gender=profile.gender,
    )
    _mesh_store[profile.user_id] = mesh_bytes

    return profile


@router.get("/{user_id}", response_model=BodyProfile)
async def get_body_profile(user_id: str):
    """Retrieve a user's body profile."""
    profile = _profiles.get(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Body profile not found")
    return profile


@router.get("/{user_id}/mesh.glb")
async def get_body_mesh(user_id: str):
    """Serve the user's 3D body mesh."""
    from fastapi.responses import Response

    mesh_bytes = _mesh_store.get(user_id)
    if not mesh_bytes:
        raise HTTPException(status_code=404, detail="Mesh not found")
    return Response(content=mesh_bytes, media_type="model/gltf-binary")


_mesh_store: dict[str, bytes] = {}
_photo_store: dict[str, str] = {}  # user_id -> front photo base64
