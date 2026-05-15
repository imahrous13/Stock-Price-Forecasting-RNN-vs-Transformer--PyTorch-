import pandas as pd
import numpy as np
import yfinance as yf
import os
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
       
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        
        if save:
            filepath = os.path.join(self.data_dir, f"{symbol}_{start_date}_{end_date}.csv")
            data.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        
        return data
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        data = pd.read_csv(filepath)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], utc=True, errors='coerce').dt.tz_localize(None)
        return data
    
    def inspect_data(self, data: pd.DataFrame) -> dict:
        inspection = {
            'shape': data.shape,
            'date_range': {
                'start': data['Date'].min() if 'Date' in data.columns else None,
                'end': data['Date'].max() if 'Date' in data.columns else None
            },
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'dtypes': data.dtypes.to_dict(),
            'statistics': data.describe().to_dict(),
            'outliers': {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            inspection['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100
            }
        
        return inspection
    
    def print_inspection(self, inspection: dict):
        print("=" * 60)
        print("DATA INSPECTION REPORT")
        print("=" * 60)
        print(f"\nDataset Shape: {inspection['shape']}")
        
        if inspection['date_range']['start']:
            print(f"\nDate Range:")
            print(f"  Start: {inspection['date_range']['start']}")
            print(f"  End: {inspection['date_range']['end']}")
        
        print(f"\nMissing Values:")
        for col, count in inspection['missing_values'].items():
            if count > 0:
                pct = inspection['missing_percentage'][col]
                print(f"  {col}: {count} ({pct:.2f}%)")
        
        print(f"\nOutliers (IQR method):")
        for col, info in inspection['outliers'].items():
            if info['count'] > 0:
                print(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")
        
        print("\n" + "=" * 60)
    
    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        df['Volatility'] = df['Returns'].rolling(window=10).std()
        
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        df = df.dropna().reset_index(drop=True)
        
        return df



