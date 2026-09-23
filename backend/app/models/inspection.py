from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Inspection(Base):
    """Una inspección: la foto analizada y sus resultados.

    `category` y `method` se agregaron en el commit 7 para poder mostrar en el
    historial si el producto lo identificó el modelo preentrenado o el
    clasificador de respaldo (y si es fruta o verdura).
    """

    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    item = Column(String(50), nullable=False)
    category = Column(String(20), nullable=False, default="Unknown")
    quality = Column(String(20), nullable=False)
    ripeness = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    method = Column(String(30), nullable=False, default="unrecognized")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="inspections")
