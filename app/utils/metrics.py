"""
Simple metric calculations as fallback
"""
import numpy as np
from typing import List

def calculate_mae(true_values: List[float], predicted_values: List[float]) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(np.array(true_values) - np.array(predicted_values)))

def calculate_rmse(true_values: List[float], predicted_values: List[float]) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((np.array(true_values) - np.array(predicted_values)) ** 2))

def calculate_r2(true_values: List[float], predicted_values: List[float]) -> float:
    """Calculate R-squared"""
    true_mean = np.mean(true_values)
    ss_total = np.sum((np.array(true_values) - true_mean) ** 2)
    ss_residual = np.sum((np.array(true_values) - np.array(predicted_values)) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    return 1 - (ss_residual / ss_total)