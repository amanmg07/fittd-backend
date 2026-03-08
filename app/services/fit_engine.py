"""
Fit prediction engine: compares user body measurements against garment
size charts to recommend the best size and predict how it will fit.

Accounts for:
- Garment measurements vs body measurements (ease/allowance)
- Fabric stretch properties
- Fit type (slim vs oversized expects different ease)
- Per-measurement scoring (chest, waist, hips, shoulders, length, sleeves)
"""

from app.models.schemas import (
    BodyMeasurements,
    GarmentInfo,
    GarmentSize,
    FitType,
    SizeRecommendation,
)


# Expected ease (garment measurement - body measurement) by fit type, in cm
EASE_TARGETS: dict[FitType, dict[str, float]] = {
    FitType.slim: {"chest": 4.0, "waist": 2.0, "hips": 2.0, "shoulder": 0.5},
    FitType.regular: {"chest": 10.0, "waist": 8.0, "hips": 6.0, "shoulder": 2.0},
    FitType.relaxed: {"chest": 16.0, "waist": 14.0, "hips": 10.0, "shoulder": 4.0},
    FitType.oversized: {"chest": 22.0, "waist": 20.0, "hips": 14.0, "shoulder": 6.0},
}

# How much each fabric type stretches (multiplier on garment measurement)
STRETCH_FACTORS: dict[str, float] = {
    "elastane": 0.15,
    "spandex": 0.15,
    "lycra": 0.15,
    "polyester": 0.02,
    "nylon": 0.03,
    "cotton": 0.01,
    "wool": 0.02,
}


def calculate_stretch_allowance(materials: dict[str, float]) -> float:
    """
    Calculate total stretch factor based on material composition.
    Returns a multiplier (e.g., 0.05 = 5% stretch).
    """
    total_stretch = 0.0
    for material, proportion in materials.items():
        material_lower = material.lower()
        stretch = STRETCH_FACTORS.get(material_lower, 0.01)
        total_stretch += stretch * proportion
    return total_stretch


def score_size(
    body: BodyMeasurements,
    garment_size: GarmentSize,
    fit_type: FitType,
    stretch: float,
) -> tuple[float, list[str]]:
    """
    Score how well a garment size fits the user.
    Returns (score 0-1, list of fit notes).
    Higher score = better fit.
    """
    targets = EASE_TARGETS[fit_type]
    notes = []
    penalties = []

    # --- Chest ---
    effective_chest = garment_size.chest_cm * (1 + stretch)
    chest_ease = effective_chest - body.chest
    target_chest_ease = targets["chest"]
    chest_diff = abs(chest_ease - target_chest_ease)

    if chest_ease < 0:
        notes.append(f"Chest will be very tight ({abs(chest_ease):.0f}cm too small)")
        penalties.append(chest_diff * 3)
    elif chest_diff < 3:
        notes.append("Chest fit is ideal")
        penalties.append(0)
    elif chest_ease > target_chest_ease + 6:
        notes.append(f"Chest will be loose ({chest_ease - target_chest_ease:.0f}cm extra)")
        penalties.append(chest_diff * 1.5)
    else:
        notes.append("Chest fit is good")
        penalties.append(chest_diff)

    # --- Waist ---
    if garment_size.waist_cm and body.waist:
        effective_waist = garment_size.waist_cm * (1 + stretch)
        waist_ease = effective_waist - body.waist
        target_waist_ease = targets["waist"]
        waist_diff = abs(waist_ease - target_waist_ease)

        if waist_ease < -2:
            notes.append(f"Waist will be very tight ({abs(waist_ease):.0f}cm too small)")
            penalties.append(waist_diff * 2.5)
        elif waist_diff < 3:
            notes.append("Waist fit is ideal")
            penalties.append(0)
        elif waist_ease > target_waist_ease + 8:
            notes.append(f"Waist will be baggy ({waist_ease - target_waist_ease:.0f}cm extra)")
            penalties.append(waist_diff * 1.2)
        else:
            notes.append("Waist fit is good")
            penalties.append(waist_diff * 0.8)

    # --- Shoulder fit ---
    if garment_size.shoulder_cm and body.shoulder_width:
        shoulder_ease = garment_size.shoulder_cm - body.shoulder_width
        target_shoulder = targets["shoulder"]
        shoulder_diff = abs(shoulder_ease - target_shoulder)

        if shoulder_ease < -1:
            notes.append("Shoulders will be tight")
            penalties.append(shoulder_diff * 2.5)
        elif shoulder_diff < 2:
            notes.append("Shoulder fit is ideal")
            penalties.append(0)
        else:
            penalties.append(shoulder_diff)

    # --- Length fit (compared to torso) ---
    if garment_size.length_cm and body.torso_length:
        length_ratio = garment_size.length_cm / body.torso_length
        if length_ratio < 1.15:
            notes.append("May be short on the torso")
            penalties.append((1.15 - length_ratio) * 20)
        elif length_ratio > 1.45:
            notes.append("Will be long — could tuck in")
            penalties.append((length_ratio - 1.45) * 10)
        else:
            notes.append("Length is ideal")

    # --- Sleeve length ---
    if garment_size.sleeve_cm and body.arm_length:
        sleeve_diff = garment_size.sleeve_cm - body.arm_length * 0.55
        if sleeve_diff < -2:
            notes.append("Sleeves may be short")
            penalties.append(abs(sleeve_diff) * 1.5)
        elif sleeve_diff > 5:
            notes.append("Sleeves will be long")
            penalties.append(sleeve_diff * 0.5)

    # Convert penalties to 0-1 score
    total_penalty = sum(penalties)
    score = max(0.0, 1.0 - (total_penalty / 30.0))

    return round(score, 3), notes


def recommend_size(
    body: BodyMeasurements,
    garment: GarmentInfo,
) -> SizeRecommendation:
    """
    Recommend the best size for a user given their measurements
    and the garment's properties.
    """
    stretch = calculate_stretch_allowance(garment.material_composition)

    size_scores: dict[str, float] = {}
    size_notes: dict[str, list[str]] = {}

    for gs in garment.sizes:
        score, notes = score_size(body, gs, garment.fit_type, stretch)
        size_scores[gs.size_label] = score
        size_notes[gs.size_label] = notes

    best_size = max(size_scores, key=size_scores.get)

    return SizeRecommendation(
        recommended_size=best_size,
        confidence=size_scores[best_size],
        fit_notes=size_notes[best_size],
        size_scores=size_scores,
    )
