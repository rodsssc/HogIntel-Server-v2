# app/routes/price.py
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from schemas import PriceRequest, PriceResponse, ErrorResponse, DataQuality
from models.price_model import PricePredictor
from logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Initialize price predictor
try:
    # Pass db_connection if available (replace None with actual connection)
    price_predictor = PricePredictor(db_connection=None)
    logger.info("✅ SARIMA price prediction model initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize price model: {e}")
    raise

# Dependency to ensure predictor is loaded
def get_price_predictor() -> PricePredictor:
    """Dependency to get price predictor instance"""
    if not price_predictor.model_loaded:
        logger.warning("⚠️  SARIMA model not loaded, using fallback mode")
    return price_predictor


@router.post(
    "/price",
    response_model=PriceResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Estimate market price for confirmed weight",
    description="Stage 2: Predict price per kg using SARIMA time series model and compute total value after weight confirmation"
)
async def predict_price(
    request: PriceRequest,
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Predict market price for confirmed hog weight.
    
    This endpoint:
    - Uses confirmed weight from previous stage
    - Predicts price per kg using SARIMA (Seasonal ARIMA) time series model
    - Considers seasonality, trends, and temporal patterns
    - Computes total value
    - Returns price prediction with market context and confidence
    
    **Model**: SARIMA (Seasonal AutoRegressive Integrated Moving Average)
    **Features**: Temporal patterns, seasonality, trends, and historical price patterns
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
        logger.warning(f"⚠️  Unusual weight detected: {request.confirmed_weight} kg")
    
    # Check if historical prices need updating
    if len(predictor.historical_prices) == 0:
        logger.warning("⚠️  No historical prices loaded! Predictions will be less accurate.")
        logger.warning("⚠️  Call /api/v1/price/update-historical to improve accuracy.")
    
    try:
        # Predict price with optional date parameter
        prediction_date = getattr(request, 'prediction_date', None)
        
        logger.info(f"🔮 Starting SARIMA price prediction for scan {request.scan_id}")
        logger.info(f"   Weight: {request.confirmed_weight} kg")
        logger.info(f"   Historical data points: {len(predictor.historical_prices)}")
        
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
            f"✅ Scan {request.scan_id}: SARIMA price prediction complete - "
            f"₱{price_prediction.price_per_kg:.2f}/kg, "
            f"Total: ₱{total_value:.2f}, "
            f"Model: {price_prediction.model_name}, "
            f"Confidence: {price_prediction.confidence:.2%}, "
            f"Forecast Horizon: {price_prediction.prediction_metadata.get('forecast_horizon', 'N/A') if price_prediction.prediction_metadata else 'N/A'}"
        )
        
        # Determine data quality from prediction metadata
        data_quality = DataQuality.GOOD  # Default to good
        if price_prediction.prediction_metadata:
            data_sufficiency = price_prediction.prediction_metadata.get('data_sufficiency', 'good')
            if data_sufficiency == 'limited':
                data_quality = DataQuality.LIMITED
            elif data_sufficiency == 'insufficient':
                data_quality = DataQuality.INSUFFICIENT
        
        # Prepare response with enhanced information - INCLUDING data_quality
        response_data = {
            "price_per_kg": price_prediction.price_per_kg,
            "total_value": total_value,
            "confidence": price_prediction.confidence,
            "model_used": price_prediction.model_name,
            "market_conditions": price_prediction.market_conditions,
            "timestamp": datetime.now(),
            "data_quality": data_quality  # CRITICAL: Add this required field
        }
        
        # Add optional metadata
        if price_prediction.features_used:
            response_data["features_used"] = price_prediction.features_used
        
        if price_prediction.prediction_metadata:
            response_data["prediction_metadata"] = price_prediction.prediction_metadata
        
        return PriceResponse(**response_data)
        
    except ValueError as e:
        logger.error(f"❌ Validation error for scan {request.scan_id}: {str(e)}")
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
        logger.error(f"❌ SARIMA price prediction failed for scan {request.scan_id}: {str(e)}", exc_info=True)
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
    description="Returns information about SARIMA model and fallback options"
)
async def get_available_models(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Get information about available price prediction models.
    
    Returns:
    - Model status (loaded/unavailable)
    - Model type and description
    - SARIMA order parameters
    - Performance metrics (if available)
    - Fallback options
    """
    try:
        models_info = predictor.get_available_models()
        
        return {
            "available_models": models_info,
            "default_model": "sarima",
            "fallback_available": predictor.fallback_available,
            "status": "operational" if predictor.model_loaded else "fallback_only"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get model info: {str(e)}")
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
    summary="Get comprehensive SARIMA model information",
    description="Returns detailed information about the SARIMA model, parameters, and configuration"
)
async def get_model_info(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Get comprehensive information about the SARIMA price prediction model.
    
    Returns:
    - Model type and status
    - SARIMA order and seasonal order parameters
    - Performance metrics (AIC, BIC, RMSE, etc.)
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
        logger.error(f"❌ Failed to get comprehensive model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve comprehensive model information",
                "code": "MODEL_DETAIL_ERROR"
            }
        )


@router.post(
    "/price/update-historical",
    summary="Update historical price data for SARIMA",
    description="Update the model's historical price data for SARIMA time series forecasting"
)
async def update_historical_prices(
    prices: List[Dict[str, Any]],
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Update historical price data for improved SARIMA predictions.
    
    Args:
        prices: List of dicts with 'price_date' (ISO string) and 'price' (float) keys
        
    Expected format:
    ```json
    [
        {"price_date": "2024-01-01", "price": 175.0},
        {"price_date": "2024-02-01", "price": 178.5}
    ]
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
        
        # Validate minimum data requirements for SARIMA
        if len(prices) < 12:
            logger.warning(f"⚠️  Limited historical data: {len(prices)} records. SARIMA works best with 24+ months of data.")
        
        # Validate each record
        validated_prices = []
        for i, record in enumerate(prices):
            if 'price_date' not in record or 'price' not in record:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid record format",
                        "code": "MISSING_FIELDS",
                        "details": {
                            "record_index": i,
                            "message": "Each record must have 'price_date' and 'price' fields"
                        }
                    }
                )
            
            # Convert date string to datetime if needed
            if isinstance(record['price_date'], str):
                try:
                    date_obj = datetime.fromisoformat(record['price_date'].replace('Z', '+00:00'))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid date format",
                            "code": "INVALID_DATE",
                            "details": {
                                "record_index": i,
                                "date": record['price_date'],
                                "message": "Date must be in ISO format"
                            }
                        }
                    )
            else:
                date_obj = record['price_date']
            
            validated_prices.append({
                'date': date_obj,  # Keep internal name as 'date' for the model
                'price': float(record['price'])
            })
        
        # Update historical prices
        predictor.update_historical_prices(validated_prices)
        
        logger.info(f"✅ Historical prices updated for SARIMA: {len(validated_prices)} records")
        
        return {
            "success": True,
            "records_updated": len(validated_prices),
            "date_range": {
                "start": validated_prices[0]['date'].isoformat() if validated_prices else None,
                "end": validated_prices[-1]['date'].isoformat() if validated_prices else None
            },
            "last_known_price": predictor.last_known_price,
            "data_sufficiency": "sufficient" if len(validated_prices) >= 24 else "limited",
            "message": f"Historical prices updated successfully. SARIMA predictions will now be more accurate. Data sufficiency: {'✓ Sufficient' if len(validated_prices) >= 24 else '⚠ Limited'}",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update historical prices: {str(e)}", exc_info=True)
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
    summary="Health check for SARIMA price prediction service",
    description="Check if the SARIMA model is loaded and operational"
)
async def health_check(predictor: PricePredictor = Depends(get_price_predictor)):
    """
    Check the health status of the SARIMA price prediction service.
    
    Returns:
    - Overall service status
    - Model loading status
    - Historical data availability and sufficiency
    - Last known price
    - SARIMA model parameters
    """
    try:
        model_info = predictor.get_model_info()
        
        # Determine health status based on SARIMA requirements
        historical_sufficient = model_info["historical_data_points"] >= 12
        
        if model_info["model_loaded"] and historical_sufficient:
            status = "healthy"
            message = "SARIMA model operational with sufficient historical data"
        elif model_info["model_loaded"]:
            status = "degraded"
            message = "SARIMA model loaded but historical data is limited"
        else:
            status = "degraded"
            message = "Using fallback mode - SARIMA model not loaded"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "model_loaded": model_info["model_loaded"],
                "historical_data_points": model_info["historical_data_points"],
                "historical_data_sufficiency": "sufficient" if historical_sufficient else "limited",
                "last_known_price": model_info["last_known_price"],
                "fallback_available": model_info["fallback_available"],
                "sarima_parameters": model_info.get("sarima_parameters", {}),
                "minimum_data_required": 12,
                "recommended_data_points": 24
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": "Service error",
            "error": str(e),
            "timestamp": datetime.now()
        }


@router.post(
    "/price/batch",
    summary="Batch SARIMA price prediction",
    description="Predict prices for multiple hogs at once using SARIMA model"
)
async def batch_predict_price(
    requests: List[PriceRequest],
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Predict prices for multiple hogs in a single request using SARIMA.
    
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
            
            # Predict using SARIMA
            price_prediction = predictor.predict(
                weight_kg=request.confirmed_weight,
                market_data=request.market_data,
                use_fallback=request.use_fallback
            )
            
            total_value = price_prediction.price_per_kg * request.confirmed_weight
            
            # Determine data quality for batch response
            data_quality = "good"
            if price_prediction.prediction_metadata:
                data_sufficiency = price_prediction.prediction_metadata.get('data_sufficiency', 'good')
                if data_sufficiency == 'limited':
                    data_quality = "limited"
                elif data_sufficiency == 'insufficient':
                    data_quality = "insufficient"
            
            results.append({
                "scan_id": request.scan_id,
                "price_per_kg": price_prediction.price_per_kg,
                "total_value": total_value,
                "confidence": price_prediction.confidence,
                "model_used": price_prediction.model_name,
                "forecast_horizon": price_prediction.prediction_metadata.get('forecast_horizon', 'N/A') if price_prediction.prediction_metadata else 'N/A',
                "data_quality": data_quality
            })
            
        except Exception as e:
            logger.error(f"❌ Batch SARIMA prediction failed for index {idx}: {str(e)}")
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


@router.get(
    "/price/forecast",
    summary="Get SARIMA price forecasts",
    description="Generate future price forecasts using the SARIMA model"
)
async def get_price_forecast(
    periods: int = 30,
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Generate future price forecasts using SARIMA model.
    
    Args:
        periods: Number of future periods to forecast (default: 30 days)
    
    Returns:
        Price forecasts with confidence intervals
    """
    if periods <= 0 or periods > 365:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid forecast period",
                "code": "INVALID_PERIOD",
                "message": "Period must be between 1 and 365 days"
            }
        )
    
    try:
        forecasts = predictor.generate_forecast(periods=periods)
        return forecasts
    except Exception as e:
        logger.error(f"❌ SARIMA forecast generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Forecast generation failed",
                "code": "FORECAST_ERROR",
                "message": str(e)
            }
        )