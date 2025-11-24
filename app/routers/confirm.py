# app/routers/confirm.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

class ConfirmRequest(BaseModel):
    scan_id: str
    confirmed_weight: float
    status: str  # accepted, rejected, adjusted
    notes: Optional[str] = None

class ConfirmResponse(BaseModel):
    success: bool
    confirmed_weight: float
    scan_id: str
    timestamp: datetime
    previous_weight: Optional[float] = None
    adjustment: Optional[float] = None

@router.post(
    "/confirm",
    response_model=ConfirmResponse,
    summary="Confirm weight prediction",
    description="Stage 1c: User confirms, rejects, or adjusts the predicted weight"
)
async def confirm_weight(request: ConfirmRequest):
    """Stage 1c: User confirms/adjusts weight"""
    try:
        # Validate status
        valid_statuses = ["accepted", "rejected", "adjusted"]
        if request.status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        # Validate weight
        if request.confirmed_weight <= 0:
            raise HTTPException(
                status_code=400,
                detail="Weight must be greater than 0"
            )
        
        # Store confirmation for feedback loop (in a real app, save to database)
        logger.info(
            f"✅ Weight {request.status}: {request.confirmed_weight}kg for scan {request.scan_id}"
        )
        
        # In a real implementation, you would:
        # 1. Store the confirmation in a database
        # 2. Update model training data
        # 3. Track user adjustments for model improvement
        
        response_data = {
            "success": True,
            "confirmed_weight": request.confirmed_weight,
            "scan_id": request.scan_id,
            "timestamp": datetime.now()
        }
        
        # Add adjustment info if status is adjusted
        if request.status == "adjusted" and request.notes:
            response_data["adjustment"] = request.confirmed_weight  # You might want to calculate actual adjustment
            logger.info(f"📝 User notes: {request.notes}")
        
        return ConfirmResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Confirmation error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Confirmation failed: {str(e)}"
        )

@router.get("/confirm/health")
async def health_check():
    """Health check for confirmation endpoint"""
    return {
        "status": "healthy",
        "service": "weight_confirmation",
        "timestamp": datetime.now()
    }