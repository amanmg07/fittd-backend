from sqlalchemy import Column, String, Float, JSON, Enum as SAEnum, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class UserBodyProfile(Base):
    __tablename__ = "user_body_profiles"

    user_id = Column(String, primary_key=True)
    gender = Column(String, nullable=False)
    height_cm = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    measurements = Column(JSON, nullable=False)
    smplx_params = Column(JSON, nullable=False)
    mesh_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class CachedGarment(Base):
    __tablename__ = "cached_garments"

    product_id = Column(String, primary_key=True)
    brand = Column(String, nullable=False)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    data = Column(JSON, nullable=False)  # full GarmentInfo as JSON
    garment_mesh_url = Column(String, nullable=True)  # 3D mesh if generated
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
