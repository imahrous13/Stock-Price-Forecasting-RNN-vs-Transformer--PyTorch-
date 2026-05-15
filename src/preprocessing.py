import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List
import pickle
import os


class Preprocessor:
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        scaler_type: str = 'minmax',
        features: Optional[List[str]] = None,
        target_col: str = 'Close'
    ):
    
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        self.features = features
        self.target_col = target_col
        
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.target_scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.is_fitted = False
        self.feature_columns = None
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Forward fill for missing values
        df = df.ffill()
        
        # If still missing, backward fill
        df = df.bfill()
        
        # If still missing, drop rows
        df = df.dropna()
        
        return df
    
    def _handle_outliers(self, data: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
     
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            
            if method == 'clip':
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df.reset_index(drop=True)
    
    def fit(self, data: pd.DataFrame):
       
        df = data.copy()
        
        # Handle missing values and outliers
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        
        # Determine feature columns
        if self.features is None:
            # Use all numeric columns except Date and target
            exclude_cols = ['Date', self.target_col] if 'Date' in df.columns else [self.target_col]
            self.feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                                  if col not in exclude_cols]
        else:
            self.feature_columns = self.features
        
        # Fit scalers
        self.scaler.fit(df[self.feature_columns])
        self.target_scaler.fit(df[[self.target_col]])
        
        self.is_fitted = True
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
   
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        df = data.copy()
        
        # Handle missing values and outliers
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df[self.feature_columns])
        y_scaled = self.target_scaler.transform(df[[self.target_col]])
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            X.append(X_scaled[i:i + self.sequence_length])
            y.append(y_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
     
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
       
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save(self, filepath: str):
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'scaler_type': self.scaler_type,
                'features': self.features,
                'target_col': self.target_col,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'is_fitted': self.is_fitted,
                'feature_columns': self.feature_columns
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Preprocessor':
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(
            sequence_length=data['sequence_length'],
            prediction_horizon=data['prediction_horizon'],
            scaler_type=data['scaler_type'],
            features=data['features'],
            target_col=data['target_col']
        )
        preprocessor.scaler = data['scaler']
        preprocessor.target_scaler = data['target_scaler']
        preprocessor.is_fitted = data['is_fitted']
        preprocessor.feature_columns = data['feature_columns']
        
        return preprocessor

