import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import StockDataLoader
from .preprocessing import Preprocessor
from .models import VanillaRNN, TransformerModel, CausalTransformerModel
from .training import Trainer, set_seed
from .evaluation import calculate_metrics, ModelComparator


SUPPORTED_MODELS = ('rnn', 'transformer', 'transformer_causal')


class Pipeline:

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        early_stopping_patience: int = 15,
        device: Optional[torch.device] = None,
        seed: int = 42,
        data_dir: str = "data/raw",
        models_dir: str = "models",
        results_dir: str = "results",
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.models_dir = models_dir
        self.results_dir = results_dir

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        set_seed(seed)

        self.data_loader = StockDataLoader(data_dir=data_dir)
        self.preprocessor: Optional[Preprocessor] = None
        self.data = None
        self.models: Dict[str, torch.nn.Module] = {}
        self.trainers: Dict[str, Trainer] = {}
        self.comparator = ModelComparator(results_dir)

        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

    def load_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_existing: bool = True,
    ):
        print("Loading data...")
        filepath = os.path.join(
            self.data_loader.data_dir,
            f"{symbol}_{start_date}_{end_date}.csv",
        )

        if use_existing and os.path.exists(filepath):
            self.data = self.data_loader.load_from_file(filepath)
            print(f"Loaded cached data from {filepath}")
        else:
            self.data = self.data_loader.fetch_data(symbol, start_date, end_date)

        self.data = self.data_loader.get_features(self.data)

        inspection = self.data_loader.inspect_data(self.data)
        self.data_loader.print_inspection(inspection)
        return self.data

    def preprocess_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: str = 'minmax',
        features: Optional[List[str]] = None,
    ):
        print("\nPreprocessing data...")
        self.preprocessor = Preprocessor(
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            scaler_type=scaler_type,
            features=features,
        )

        X, y = self.preprocessor.fit_transform(self.data)

        (
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        ) = self.preprocessor.split_data(X, y, train_ratio, val_ratio)

        print(f"Training samples:   {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples:       {len(self.X_test)}")
        print(f"Input shape:        {self.X_train.shape}")
        print(f"Output shape:       {self.y_train.shape}")

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        X_train_t = torch.FloatTensor(self.X_train)
        y_train_t = torch.FloatTensor(self.y_train)
        X_val_t   = torch.FloatTensor(self.X_val)
        y_val_t   = torch.FloatTensor(self.y_val)
        X_test_t  = torch.FloatTensor(self.X_test)
        y_test_t  = torch.FloatTensor(self.y_test)

        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds   = TensorDataset(X_val_t,   y_val_t)
        test_ds  = TensorDataset(X_test_t,  y_test_t)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _build_model(self, model_type: str, input_size: int, output_size: int) -> torch.nn.Module:
        if model_type == 'rnn':
            return VanillaRNN(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout,
                use_batch_norm=self.use_batch_norm,
            )
        if model_type in ('transformer', 'transformer_causal'):
            nhead = 8 if self.hidden_size % 8 == 0 else (
                4 if self.hidden_size % 4 == 0 else 2
            )
            if model_type == 'transformer':
                return TransformerModel(
                    input_size=input_size,
                    d_model=self.hidden_size,
                    nhead=nhead,
                    num_layers=self.num_layers,
                    dim_feedforward=self.hidden_size * 4,
                    output_size=output_size,
                    dropout=self.dropout,
                    use_batch_norm=False,
                )
            return CausalTransformerModel(
                input_size=input_size,
                d_model=self.hidden_size,
                nhead=nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.hidden_size * 4,
                output_size=output_size,
                dropout=self.dropout,
                max_seq_len=max(256, self.sequence_length + 16),
            )
        raise ValueError(
            f"Unsupported model '{model_type}'. Supported: {SUPPORTED_MODELS}"
        )

    def train_models(self, model_types: Optional[List[str]] = None):
        if self.preprocessor is None:
            raise ValueError("Data must be preprocessed before training")

        if model_types is None:
            model_types = list(SUPPORTED_MODELS)

        unknown = [m for m in model_types if m not in SUPPORTED_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}. Supported: {SUPPORTED_MODELS}"
            )

        input_size = self.X_train.shape[2]
        output_size = self.prediction_horizon

        train_loader, val_loader, test_loader = self.create_data_loaders()

        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*60}")

            model = self._build_model(model_type, input_size, output_size)

            trainer = Trainer(
                model=model,
                device=self.device,
                learning_rate=self.learning_rate,
                early_stopping_patience=self.early_stopping_patience,
            )

            history = trainer.train(train_loader, val_loader, self.num_epochs)

            y_pred = trainer.predict(test_loader)
            y_true = self.y_test

            y_pred_original = self.preprocessor.inverse_transform_target(y_pred)
            y_true_original = self.preprocessor.inverse_transform_target(y_true)

            metrics = calculate_metrics(y_true_original, y_pred_original)

            self.models[model_type] = model
            self.trainers[model_type] = trainer

            training_time = history.get('training_time', 0.0)
            self.comparator.add_model(
                model_type.upper(),
                y_true_original,
                y_pred_original,
                training_time,
                history,
            )

            model_path = os.path.join(self.models_dir, f"{model_type}_model.pt")
            trainer.save_model(model_path)

            print(f"\n{model_type.upper()} Test Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")

    def evaluate_and_compare(self):
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON REPORT (RNN vs TRANSFORMER)")
        print("=" * 60)

        self.comparator.generate_report(
            save_path=os.path.join(self.results_dir, 'comparison_report.csv')
        )

    def predict(
        self,
        model_type: str,
        X: np.ndarray,
        return_original_scale: bool = True,
    ) -> np.ndarray:
        if model_type not in self.trainers:
            raise ValueError(
                f"Model '{model_type}' not found. Train the model first. "
                f"Available: {list(self.trainers.keys())}"
            )

        trainer = self.trainers[model_type]
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor), self.prediction_horizon))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        y_pred = trainer.predict(loader)

        if return_original_scale and self.preprocessor:
            y_pred = self.preprocessor.inverse_transform_target(y_pred)

        return y_pred

    def save_config(self, filepath: str):
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'early_stopping_patience': self.early_stopping_patience,
            'supported_models': list(SUPPORTED_MODELS),
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)
