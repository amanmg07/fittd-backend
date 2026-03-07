from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import GarmentInfo
from app.scraping.nike import scrape_nike_product

router = APIRouter()

# In-memory cache for development
_garment_cache: dict[str, GarmentInfo] = {}


@router.post("/scrape", response_model=GarmentInfo)
async def scrape_garment(
    url: str,
    gender: str = Query(default="male", regex="^(male|female)$"),
):
    """Scrape a garment from a Nike product URL."""
    if "nike.com" not in url:
        raise HTTPException(status_code=400, detail="Only Nike URLs are supported currently")

    try:
        garment = await scrape_nike_product(url, gender)
        _garment_cache[garment.product_id] = garment
        return garment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.get("/{product_id}", response_model=GarmentInfo)
async def get_garment(product_id: str):
    """Retrieve cached garment info."""
    garment = _garment_cache.get(product_id)
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found. Scrape it first.")
    return garment


@router.get("/", response_model=list[GarmentInfo])
async def list_garments():
    """List all cached garments."""
    return list(_garment_cache.values())
