# app/routes/price.py
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Optional, List, Dict, Any

from schemas import PriceRequest, PriceResponse, ErrorResponse
from models.price_model import PricePredictor
from logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Initialize price predictor
try:
    price_predictor = PricePredictor()
    logger.info("Ridge Regression price prediction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load price model: {e}")
    raise

# Dependency to ensure predictor is loaded
def get_price_predictor() -> PricePredictor:
    """Dependency to get price predictor instance"""
    if not price_predictor.model_loaded:
        logger.warning("Ridge model not loaded, using fallback mode")
    return price_predictor


@router.post(
    "/price",
    response_model=PriceResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Estimate market price for confirmed weight",
    description="Stage 2: Predict price per kg using Ridge Regression and compute total value after weight confirmation"
)
async def predict_price(
    request: PriceRequest,
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Predict market price for confirmed hog weight.
    
    This endpoint:
    - Uses confirmed weight from previous stage
    - Predicts price per kg using Ridge Regression with feature engineering
    - Considers historical price trends and temporal patterns
    - Computes total value
    - Returns price prediction with market context and confidence
    
    **Model**: Ridge Regression with L2 regularization
    **Features**: Temporal patterns, price lags, rolling statistics, YoY changes
    """
    
    # Validate weight
    if request.confirmed_weight <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid confirmed weight",
                "code": "INVALID_WEIGHT",
                "details": {
                    "weight": request.confirmed_weight,
                    "message": "Weight must be greater than 0"
                }
            }
        )
    
    # Validate weight range (reasonable hog weights)
    if request.confirmed_weight < 20 or request.confirmed_weight > 200:
        logger.warning(f"Unusual weight detected: {request.confirmed_weight} kg")
    
    try:
        # Predict price with optional date parameter
        prediction_date = getattr(request, 'prediction_date', None)
        
        price_prediction = predictor.predict(
            weight_kg=request.confirmed_weight,
            market_data=request.market_data,
            use_fallback=request.use_fallback,
            current_date=prediction_date
        )
        
        # Compute total value
        total_value = price_prediction.price_per_kg * request.confirmed_weight
        
        # Log prediction details
        logger.info(
            f"Scan {request.scan_id}: Price prediction - "
            f"₱{price_prediction.price_per_kg:.2f}/kg, "
            f"Total: ₱{total_value:.2f}, "
            f"Model: {price_prediction.model_used}, "
            f"Confidence: {price_prediction.confidence:.2%}, "
            f"Features: {len(price_prediction.features_used) if price_prediction.features_used else 0}"
        )
        
        # Prepare response with enhanced information
        response_data = {
            "price_per_kg": price_prediction.price_per_kg,
            "total_value": total_value,
            "confidence": price_prediction.confidence,
            "model_used": price_prediction.model_used,
            "market_conditions": price_prediction.market_conditions,
            "timestamp": datetime.now()
        }
        
        # Add optional metadata
        if price_prediction.features_used:
            response_data["features_used"] = price_prediction.features_used
        
        if price_prediction.prediction_metadata:
            response_data["metadata"] = price_prediction.prediction_metadata
        
        return PriceResponse(**response_data)
        
    except ValueError as e:
        logger.error(f"Validation error for scan {request.scan_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input data",
                "code": "VALIDATION_ERROR",
                "details": {
                    "scan_id": request.scan_id,
                    "weight": request.confirmed_weight,
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Price prediction failed for scan {request.scan_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Price prediction failed",
                "code": "PRICE_PREDICTION_ERROR",
                "details": {
                    "scan_id": request.scan_id,
                    "weight": request.confirmed_weight,
                    "message": "An error occurred during price prediction. Please try again."
                }
            }
        )


@router.get(
    "/price/models",
    summary="Get available price prediction models",
    description="Returns information about Ridge Regression model and fallback options"
)
async def get_available_models(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Get information about available price prediction models.
    
    Returns:
    - Model status (loaded/unavailable)
    - Model type and description
    - Number of features used
    - Performance metrics (if available)
    - Fallback options
    """
    try:
        models_info = predictor.get_available_models()
        
        return {
            "available_models": models_info,
            "default_model": "ridge_regression",
            "fallback_available": predictor.fallback_available,
            "status": "operational" if predictor.model_loaded else "fallback_only"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve model information",
                "code": "MODEL_INFO_ERROR",
                "message": str(e)
            }
        )


@router.get(
    "/price/model-info",
    summary="Get comprehensive model information",
    description="Returns detailed information about the Ridge Regression model, features, and configuration"
)
async def get_model_info(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Get comprehensive information about the price prediction model.
    
    Returns:
    - Model type and status
    - Selected features list
    - Performance metrics (MAE, MAPE, R², etc.)
    - Historical data statistics
    - Configuration details
    """
    try:
        model_info = predictor.get_model_info()
        
        return {
            "model": model_info,
            "timestamp": datetime.now(),
            "status": "healthy" if model_info["model_loaded"] else "degraded"
        }
    except Exception as e:
        logger.error(f"Failed to get comprehensive model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve comprehensive model information",
                "code": "MODEL_DETAIL_ERROR"
            }
        )


@router.post(
    "/price/update-historical",
    summary="Update historical price data",
    description="Update the model's historical price data for better feature engineering"
)
async def update_historical_prices(
    prices: List[Dict[str, Any]],
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Update historical price data for improved predictions.
    
    Args:
        prices: List of dicts with 'date' (ISO string) and 'price' (float) keys
        
    Expected format:
    ```json
    [
        {"date": "2024-01-01", "price": 175.0},
        {"date": "2024-02-01", "price": 178.5}
    ]
    ```
    """
    try:
        # Validate input
        if not prices or not isinstance(prices, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid input format",
                    "code": "INVALID_HISTORICAL_DATA",
                    "message": "Expected a list of price records"
                }
            )
        
        # Validate each record
        validated_prices = []
        for i, record in enumerate(prices):
            if 'date' not in record or 'price' not in record:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid record format",
                        "code": "MISSING_FIELDS",
                        "details": {
                            "record_index": i,
                            "message": "Each record must have 'date' and 'price' fields"
                        }
                    }
                )
            
            # Convert date string to datetime if needed
            if isinstance(record['date'], str):
                try:
                    date_obj = datetime.fromisoformat(record['date'].replace('Z', '+00:00'))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid date format",
                            "code": "INVALID_DATE",
                            "details": {
                                "record_index": i,
                                "date": record['date'],
                                "message": "Date must be in ISO format"
                            }
                        }
                    )
            else:
                date_obj = record['date']
            
            validated_prices.append({
                'date': date_obj,
                'price': float(record['price'])
            })
        
        # Update historical prices
        predictor.update_historical_prices(validated_prices)
        
        logger.info(f"Historical prices updated: {len(validated_prices)} records")
        
        return {
            "success": True,
            "records_updated": len(validated_prices),
            "date_range": {
                "start": validated_prices[0]['date'].isoformat() if validated_prices else None,
                "end": validated_prices[-1]['date'].isoformat() if validated_prices else None
            },
            "last_known_price": predictor.last_known_price,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update historical prices: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update historical prices",
                "code": "UPDATE_ERROR",
                "message": str(e)
            }
        )


@router.get(
    "/price/health",
    summary="Health check for price prediction service",
    description="Check if the Ridge Regression model is loaded and operational"
)
async def health_check(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Check the health status of the price prediction service.
    
    Returns:
    - Overall service status
    - Model loading status
    - Historical data availability
    - Last known price
    """
    try:
        model_info = predictor.get_model_info()
        
        # Determine health status
        if model_info["model_loaded"] and model_info["scaler_loaded"]:
            status = "healthy"
            message = "All systems operational"
        elif model_info["model_loaded"]:
            status = "degraded"
            message = "Model loaded but scaler missing"
        else:
            status = "degraded"
            message = "Using fallback mode"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "model_loaded": model_info["model_loaded"],
                "scaler_loaded": model_info["scaler_loaded"],
                "features_loaded": model_info["selected_features"] is not None,
                "historical_data_points": model_info["historical_data_points"],
                "last_known_price": model_info["last_known_price"],
                "fallback_available": model_info["fallback_available"]
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": "Service error",
            "error": str(e),
            "timestamp": datetime.now()
        }


@router.post(
    "/price/batch",
    summary="Batch price prediction",
    description="Predict prices for multiple hogs at once"
)
async def batch_predict_price(
    requests: List[PriceRequest],
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Predict prices for multiple hogs in a single request.
    
    Useful for processing multiple hogs simultaneously.
    """
    if not requests or len(requests) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Empty batch request",
                "code": "EMPTY_BATCH"
            }
        )
    
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Batch size too large",
                "code": "BATCH_TOO_LARGE",
                "message": "Maximum 100 predictions per batch"
            }
        )
    
    results = []
    errors = []
    
    for idx, request in enumerate(requests):
        try:
            # Validate weight
            if request.confirmed_weight <= 0:
                errors.append({
                    "index": idx,
                    "scan_id": request.scan_id,
                    "error": "Invalid weight"
                })
                continue
            
            # Predict
            price_prediction = predictor.predict(
                weight_kg=request.confirmed_weight,
                market_data=request.market_data,
                use_fallback=request.use_fallback
            )
            
            total_value = price_prediction.price_per_kg * request.confirmed_weight
            
            results.append({
                "scan_id": request.scan_id,
                "price_per_kg": price_prediction.price_per_kg,
                "total_value": total_value,
                "confidence": price_prediction.confidence,
                "model_used": price_prediction.model_used
            })
            
        except Exception as e:
            logger.error(f"Batch prediction failed for index {idx}: {str(e)}")
            errors.append({
                "index": idx,
                "scan_id": request.scan_id,
                "error": str(e)
            })
    
    return {
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
        "timestamp": datetime.now()
    }