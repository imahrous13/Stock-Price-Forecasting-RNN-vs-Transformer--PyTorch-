import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

from .metrics import calculate_metrics, print_metrics


class ModelComparator:
    
    def __init__(self, results_dir: str = "results"):
    
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}
    
    def add_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        training_time: float = 0.0,
        history: Optional[Dict] = None
    ):
      
        metrics = calculate_metrics(y_true, y_pred)
        metrics['training_time'] = training_time
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'history': history
        }
    
    def compare_metrics(self) -> pd.DataFrame:
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = result['metrics'].copy()
            row['model'] = model_name
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('model')
        
        return df
    
    def plot_loss_convergence(self, save_path: Optional[str] = None):
       
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1 = axes[0]
        for model_name, result in self.results.items():
            if result['history'] and 'train_loss' in result['history']:
                train_loss = result['history']['train_loss']
                ax1.plot(train_loss, label=f"{model_name} (Train)", alpha=0.7)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for model_name, result in self.results.items():
            if result['history'] and 'val_loss' in result['history']:
                val_loss = result['history']['val_loss']
                ax2.plot(val_loss, label=f"{model_name} (Val)", alpha=0.7)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss convergence plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'loss_convergence.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_predictions(self, save_path: Optional[str] = None, num_samples: int = 200):
      
        n_models = len(self.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 5 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            y_true = result['y_true'].flatten()
            y_pred = result['y_pred'].flatten()
            
            plot_indices = np.linspace(0, len(y_true) - 1, min(num_samples, len(y_true)), dtype=int)
            
            ax.plot(plot_indices, y_true[plot_indices], label='Actual', alpha=0.7, linewidth=2)
            ax.plot(plot_indices, y_pred[plot_indices], label='Predicted', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.set_title(f'{model_name} - Predictions vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None):
      
        df = self.compare_metrics()
        
        metrics_to_plot = ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Value')
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_training_time_comparison(self, save_path: Optional[str] = None):
       
        training_times = {name: result['metrics']['training_time'] 
                         for name, result in self.results.items()}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(training_times.keys())
        times = list(training_times.values())
        
        ax.bar(models, times)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_xlabel('Model')
        ax.set_title('Training Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training time comparison plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, save_path: Optional[str] = None):
     
        print("\n" + "=" * 60)
        print("MODEL COMPARISON REPORT")
        print("=" * 60)
        
        df = self.compare_metrics()
        print("\nMetrics Comparison:")
        print(df.to_string())
        
        for model_name, result in self.results.items():
            print_metrics(result['metrics'], model_name)
        
        self.plot_loss_convergence()
        self.plot_predictions()
        self.plot_metrics_comparison()
        self.plot_training_time_comparison()
        
        if save_path:
            df.to_csv(save_path)
            print(f"\nComparison report saved to {save_path}")
        else:
            report_path = os.path.join(self.results_dir, 'comparison_report.csv')
            df.to_csv(report_path)
            print(f"\nComparison report saved to {report_path}")
        
        print("\n" + "=" * 60)



