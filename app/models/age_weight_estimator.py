# app/models/age_weight_estimator.py
from typing import Dict, Tuple, Optional
from datetime import datetime
from logger import setup_logger

logger = setup_logger(__name__)

class AgeWeightEstimator:
    """
    Estimates hog weight based on age using standard growth curves.
    Based on typical commercial pig growth patterns.
    NOW WITH 10% WEIGHT BOOST FOR ESTIMATED WEIGHTS
    """
    
    def __init__(self):
        # Age-based weight ranges (age_months: (min_kg, max_kg, average_kg))
        # Based on standard commercial pig growth rates
        self.age_weight_chart = {
            1: (7, 12, 9.5),       # 1 month: 7-12 kg
            2: (18, 25, 21.5),     # 2 months: 18-25 kg
            3: (30, 45, 37.5),     # 3 months: 30-45 kg
            4: (45, 60, 52.5),     # 4 months: 45-60 kg
            5: (60, 80, 70),       # 5 months: 60-80 kg → boosted to 77 kg avg
            6: (75, 100, 87.5),    # 6 months: 75-100 kg → boosted to 96.25 kg avg
            7: (90, 115, 102.5),   # 7 months: 90-115 kg
            8: (100, 130, 115),    # 8 months: 100-130 kg
            9: (110, 140, 125),    # 9 months: 110-140 kg
            10: (115, 150, 132.5), # 10 months: 115-150 kg
            11: (120, 160, 140),   # 11 months: 120-160 kg
            12: (125, 170, 147.5), # 12 months: 125-170 kg
        }
        
        # Gender factors (optional, only used if provided)
        self.gender_factors = {
            "male": 1.05,      # Boars typically heavier
            "female": 0.95,    # Gilts typically lighter
            "castrated": 1.0,  # Barrows (castrated males)
            "unknown": 1.0,
        }
        
        # 🆕 NEW: Weight boost factor (10%)
        self.weight_boost_factor = 1.10  # Add 10% to estimated weights
        
    def estimate_weight_from_age(
        self, 
        age_months: float,
        gender: str = "unknown"
    ) -> Dict:
        """
        Estimate weight based on age.
        🆕 NOW INCLUDES 10% WEIGHT BOOST FOR ESTIMATED WEIGHTS
        Gender is optional and only used if provided.
        
        Args:
            age_months: Age in months (can be decimal, e.g., 3.5)
            gender: Gender for weight adjustment (optional)
            
        Returns:
            Dictionary with weight estimates and metadata
        """
        try:
            # Validate age
            if age_months < 0.5:
                raise ValueError("Age must be at least 0.5 months (2 weeks)")
            if age_months > 24:
                logger.warning(f"Age {age_months} months exceeds typical market age")
            
            # Get gender factor (optional)
            gender_factor = self.gender_factors.get(gender.lower(), 1.0)
            
            # Calculate weight based on age
            if age_months <= 12:
                # Use direct lookup or interpolation for ages up to 12 months
                weight_range = self._interpolate_weight(age_months)
            else:
                # Extrapolate for older pigs (growth slows)
                base_weight = self.age_weight_chart[12]
                months_beyond = age_months - 12
                # Assume 3-5 kg per month after 12 months (slower growth)
                extra_min = months_beyond * 3
                extra_max = months_beyond * 5
                extra_avg = months_beyond * 4
                weight_range = (
                    base_weight[0] + extra_min,
                    base_weight[1] + extra_max,
                    base_weight[2] + extra_avg
                )
            
            # Apply gender adjustments (if gender is provided)
            min_weight = weight_range[0] * gender_factor
            max_weight = weight_range[1] * gender_factor
            avg_weight = weight_range[2] * gender_factor
            
            # 🆕 NEW: Apply 10% boost to estimated weights
            boosted_min_weight = min_weight * self.weight_boost_factor
            boosted_max_weight = max_weight * self.weight_boost_factor
            boosted_avg_weight = avg_weight * self.weight_boost_factor
            
            # Calculate confidence based on age
            confidence = self._calculate_confidence(age_months)
            
            result = {
                "estimated_weight_kg": {
                    "minimum": round(boosted_min_weight, 1),
                    "maximum": round(boosted_max_weight, 1),
                    "average": round(boosted_avg_weight, 1),
                    # Include original (unboosted) for reference
                    "original_average": round(avg_weight, 1),
                    "boost_applied": True,
                    "boost_percentage": 10
                },
                "age_months": age_months,
                "gender": gender if gender != "unknown" else None,
                "confidence": confidence,
                "gender_factor": gender_factor if gender != "unknown" else None,
                "growth_stage": self._classify_growth_stage(age_months),
                "market_ready": age_months >= 5 and boosted_avg_weight >= 60,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Age-based estimation (WITH 10% BOOST): {age_months}mo → "
                f"{boosted_min_weight:.1f}-{boosted_max_weight:.1f}kg (avg: {boosted_avg_weight:.1f}kg)"
                f" [Original avg: {avg_weight:.1f}kg]"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Age weight estimation error: {e}")
            raise
    
    def _interpolate_weight(self, age_months: float) -> Tuple[float, float, float]:
        """
        Interpolate weight for ages between chart entries.
        
        Args:
            age_months: Age in months (can be decimal)
            
        Returns:
            Tuple of (min_weight, max_weight, avg_weight)
        """
        # Round down to get lower bound
        lower_age = int(age_months)
        upper_age = lower_age + 1
        
        # Handle edge cases
        if lower_age < 1:
            lower_age = 1
            upper_age = 1
        if lower_age > 11:
            lower_age = 12
            upper_age = 12
        
        # Get weight ranges
        lower_weights = self.age_weight_chart.get(lower_age, self.age_weight_chart[1])
        upper_weights = self.age_weight_chart.get(upper_age, self.age_weight_chart[12])
        
        # Linear interpolation
        fraction = age_months - lower_age
        
        min_weight = lower_weights[0] + (upper_weights[0] - lower_weights[0]) * fraction
        max_weight = lower_weights[1] + (upper_weights[1] - lower_weights[1]) * fraction
        avg_weight = lower_weights[2] + (upper_weights[2] - lower_weights[2]) * fraction
        
        return (min_weight, max_weight, avg_weight)
    
    def _calculate_confidence(self, age_months: float) -> float:
        """
        Calculate confidence score based on age.
        Higher confidence for typical market ages (4-7 months).
        
        Args:
            age_months: Age in months
            
        Returns:
            Confidence score between 0 and 1
        """
        # Peak confidence at 5-6 months (typical market age)
        if 4 <= age_months <= 7:
            return 0.90
        elif 3 <= age_months <= 8:
            return 0.85
        elif 2 <= age_months <= 9:
            return 0.80
        elif 1 <= age_months <= 10:
            return 0.75
        elif age_months < 1:
            return 0.65  # Less certain for very young pigs
        else:
            # Decreasing confidence for older pigs
            return max(0.60, 0.75 - (age_months - 10) * 0.02)
    
    def _classify_growth_stage(self, age_months: float) -> str:
        """
        Classify the pig's growth stage based on age.
        
        Args:
            age_months: Age in months
            
        Returns:
            Growth stage classification
        """
        if age_months < 1:
            return "nursing"
        elif age_months < 2.5:
            return "weaning"
        elif age_months < 4:
            return "grower"
        elif age_months < 7:
            return "finisher"
        else:
            return "market_ready"
    
    def compare_with_image_prediction(
        self,
        age_months: float,
        image_predicted_weight: float,
        gender: str = "unknown"
    ) -> Dict:
        """
        Compare age-based estimation with image-based prediction.
        Provides validation and flags outliers.
        
        Args:
            age_months: Age in months
            image_predicted_weight: Weight predicted from image (kg)
            gender: Gender (optional)
            
        Returns:
            Comparison analysis
        """
        try:
            # Get age-based estimate (with 10% boost already applied)
            age_estimate = self.estimate_weight_from_age(age_months, gender)
            
            min_weight = age_estimate["estimated_weight_kg"]["minimum"]
            max_weight = age_estimate["estimated_weight_kg"]["maximum"]
            avg_weight = age_estimate["estimated_weight_kg"]["average"]
            
            # Check if image prediction falls within age-based range
            within_range = min_weight <= image_predicted_weight <= max_weight
            
            # Calculate deviation from average
            deviation_kg = image_predicted_weight - avg_weight
            deviation_percent = (deviation_kg / avg_weight) * 100
            
            # Determine agreement level
            if within_range:
                if abs(deviation_percent) < 10:
                    agreement = "excellent"
                elif abs(deviation_percent) < 20:
                    agreement = "good"
                else:
                    agreement = "acceptable"
            else:
                if image_predicted_weight < min_weight:
                    agreement = "underweight_concern"
                else:
                    agreement = "overweight_concern"
            
            # Calculate blended weight (weighted average)
            image_confidence = 0.85  # From CNN model
            age_confidence = age_estimate["confidence"]
            
            total_confidence = image_confidence + age_confidence
            blended_weight = (
                (image_predicted_weight * image_confidence + avg_weight * age_confidence) 
                / total_confidence
            )
            
            result = {
                "age_based_estimate": age_estimate["estimated_weight_kg"],
                "image_predicted_weight": round(image_predicted_weight, 1),
                "blended_weight": round(blended_weight, 1),
                "within_expected_range": within_range,
                "deviation_kg": round(deviation_kg, 1),
                "deviation_percent": round(deviation_percent, 1),
                "agreement_level": agreement,
                "recommendation": self._generate_recommendation(
                    agreement, deviation_kg, age_months
                ),
                "confidence_scores": {
                    "image_prediction": image_confidence,
                    "age_estimation": age_confidence,
                    "blended": round((image_confidence + age_confidence) / 2, 2)
                }
            }
            
            logger.info(
                f"Weight comparison: Image={image_predicted_weight:.1f}kg, "
                f"Age-based (boosted)={avg_weight:.1f}kg, Agreement={agreement}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Weight comparison error: {e}")
            raise
    
    def _generate_recommendation(
        self, 
        agreement: str, 
        deviation_kg: float,
        age_months: float
    ) -> str:
        """Generate recommendation based on comparison results."""
        if agreement == "excellent":
            return "Weight prediction is highly consistent with expected age-based weight."
        elif agreement == "good":
            return "Weight prediction aligns well with age-based estimates."
        elif agreement == "acceptable":
            return "Weight is within expected range but shows some variation from typical growth."
        elif agreement == "underweight_concern":
            return (
                f"Pig appears underweight for {age_months:.1f} months old. "
                "Consider nutrition assessment or health check."
            )
        else:  # overweight_concern
            return (
                f"Pig appears heavier than typical for {age_months:.1f} months old. "
                "This may indicate good growth or early market readiness."
            )
    
    def get_growth_recommendations(self, age_months: float, current_weight: float) -> Dict:
        """
        Provide growth recommendations based on current age and weight.
        
        Args:
            age_months: Current age in months
            current_weight: Current weight in kg
            
        Returns:
            Growth recommendations and projections
        """
        try:
            age_estimate = self.estimate_weight_from_age(age_months)
            expected_avg = age_estimate["estimated_weight_kg"]["average"]
            
            # Calculate growth rate compared to average
            weight_ratio = current_weight / expected_avg
            
            # Project market ready date (assuming target weight of 100kg)
            target_weight = 100
            if current_weight >= target_weight:
                days_to_market = 0
                market_ready = True
            else:
                # Estimate daily gain (typical: 0.6-0.8 kg/day)
                avg_daily_gain = 0.7
                weight_needed = target_weight - current_weight
                days_to_market = int(weight_needed / avg_daily_gain)
                market_ready = False
            
            # Growth status
            if weight_ratio >= 1.15:
                growth_status = "above_average"
            elif weight_ratio >= 0.95:
                growth_status = "on_target"
            elif weight_ratio >= 0.85:
                growth_status = "slightly_below"
            else:
                growth_status = "below_target"
            
            return {
                "current_status": {
                    "age_months": age_months,
                    "current_weight": current_weight,
                    "expected_weight": expected_avg,
                    "weight_ratio": round(weight_ratio, 2),
                    "growth_status": growth_status
                },
                "market_projection": {
                    "target_weight": target_weight,
                    "weight_remaining": max(0, target_weight - current_weight),
                    "estimated_days_to_market": days_to_market,
                    "market_ready": market_ready,
                    "estimated_market_date": (
                        datetime.now().timestamp() + (days_to_market * 86400)
                    )
                },
                "recommendations": self._generate_growth_recommendations(
                    growth_status, age_months, current_weight
                )
            }
            
        except Exception as e:
            logger.error(f"Growth recommendations error: {e}")
            raise
    
    def _generate_growth_recommendations(
        self, 
        growth_status: str, 
        age_months: float,
        current_weight: float
    ) -> list:
        """Generate specific growth recommendations."""
        recommendations = []
        
        if growth_status == "above_average":
            recommendations.extend([
                "Pig is growing well above average",
                "Consider early marketing if weight targets are met",
                "Monitor feed efficiency to optimize costs"
            ])
        elif growth_status == "on_target":
            recommendations.extend([
                "Growth is on track with expectations",
                "Continue current feeding program",
                "Regular weight monitoring recommended"
            ])
        elif growth_status == "slightly_below":
            recommendations.extend([
                "Growth is slightly below average",
                "Review feed quality and quantity",
                "Check for health issues or stress factors"
            ])
        else:  # below_target
            recommendations.extend([
                "Growth is significantly below target",
                "Immediate veterinary consultation recommended",
                "Review feeding program and housing conditions",
                "Check for parasites or disease"
            ])
        
        return recommendations