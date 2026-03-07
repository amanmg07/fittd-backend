from pydantic import BaseModel
from enum import Enum


class Gender(str, Enum):
    male = "male"
    female = "female"


class FitType(str, Enum):
    slim = "slim"
    regular = "regular"
    relaxed = "relaxed"
    oversized = "oversized"


class BodyMeasurements(BaseModel):
    """Measurements in centimeters."""
    height: float
    weight: float
    chest: float
    waist: float
    hips: float
    shoulder_width: float
    arm_length: float
    neck: float
    torso_length: float


class BodyScanRequest(BaseModel):
    user_id: str
    gender: Gender
    height_cm: float
    weight_kg: float
    front_image: str  # base64 encoded
    side_image: str   # base64 encoded


class BodyProfile(BaseModel):
    user_id: str
    gender: Gender
    measurements: BodyMeasurements
    mesh_url: str  # S3 URL to the SMPL-X mesh (.obj/.glb)
    smplx_params: dict  # SMPL-X shape/pose parameters for reconstruction


class BodyRestoreRequest(BaseModel):
    profile: BodyProfile
    front_photo: str | None = None  # base64 encoded front photo


class GarmentSize(BaseModel):
    size_label: str          # "S", "M", "L", "XL", etc.
    chest_cm: float
    waist_cm: float | None = None
    length_cm: float
    shoulder_cm: float | None = None
    sleeve_cm: float | None = None


class GarmentInfo(BaseModel):
    product_id: str
    name: str
    brand: str
    url: str
    image_urls: list[str]
    fit_type: FitType
    material_composition: dict[str, float]  # e.g. {"polyester": 0.88, "elastane": 0.12}
    sizes: list[GarmentSize]
    category: str  # "top"
    color: str


class SizeRecommendation(BaseModel):
    recommended_size: str
    confidence: float  # 0-1
    fit_notes: list[str]  # e.g. ["Chest will be snug", "Length is ideal"]
    size_scores: dict[str, float]  # score per size


class TryOnRequest(BaseModel):
    user_id: str
    product_id: str
    size: str | None = None  # if None, auto-recommend
    photo: str | None = None  # base64 encoded photo for AI try-on


class TryOnResult(BaseModel):
    user_id: str
    product_id: str
    selected_size: str
    recommendation: SizeRecommendation
    scene_url: str  # URL to the 3D scene file (.glb) with garment on body
    fit_map_url: str  # URL to a texture showing tight/loose areas
