# app/routers/confirm.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

class ConfirmWeightRequest(BaseModel):
    detection_id: str
    confirmed_weight: float
    user_adjusted: bool = False

@router.post("/confirm_weight")
async def confirm_weight(request: ConfirmWeightRequest):
    """Stage 1c: User confirms/adjusts weight"""
    try:
        # Store confirmation for feedback loop
        logger.info(f"Weight confirmed: {request.confirmed_weight}kg for {request.detection_id}")
        
        return {
            "success": True,
            "detection_id": request.detection_id,
            "confirmed_weight": request.confirmed_weight,
            "message": "Weight confirmed successfully"
        }
    except Exception as e:
        logger.error(f"Confirmation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))