import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'directional_accuracy': np.nan
        }
    
    mse = mean_squared_error(y_true, y_pred)
    
    rmse = np.sqrt(mse)
    
    mae = mean_absolute_error(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
  
    print(f"\n{model_name} Metrics:")
    print("=" * 50)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R²:   {metrics['r2']:.6f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print("=" * 50)



