from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import body, garments, tryon

settings = get_settings()

app = FastAPI(title="Fittd API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(body.router, prefix="/api/body", tags=["body"])
app.include_router(garments.router, prefix="/api/garments", tags=["garments"])
app.include_router(tryon.router, prefix="/api/tryon", tags=["tryon"])


@app.get("/health")
async def health():
    return {"status": "ok"}
