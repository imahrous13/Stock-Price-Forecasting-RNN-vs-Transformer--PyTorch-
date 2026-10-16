# Stock Price Forecasting: RNN vs. Transformer (PyTorch)

## Overview

A coursework/research project comparing recurrent and attention-based models for forecasting stock closing prices from historical OHLCV (open, high, low, close, volume) data and engineered features. It includes a reusable Python pipeline, an exploratory notebook, and a Streamlit interface for training and comparing models.

The repository contains cached AAPL, MSFT, and GOOGL datasets for the requested range `2020-01-01` to `2024-01-01`, model checkpoints, and comparison artifacts. The notebook uses AAPL as the target series and MSFT/GOOGL for a transfer-learning experiment.

## Features

- Download prices through `yfinance` or reuse local CSV files.
- Engineer returns, price ratios, moving averages, rolling volatility, and volume features.
- Compare four training configurations: vanilla RNN, Transformer encoder, causal Transformer, and a pretrained/fine-tuned Transformer.
- Configure sequence length, forecast horizon, model size, and training parameters.
- Export regression metrics, training times, loss curves, and prediction plots.
- Use the Streamlit UI to load data, train models, compare results, and predict from a test window or manually entered features.

## Tech Stack

- **Language/modeling:** Python, PyTorch, NumPy.
- **Data/preprocessing:** pandas, scikit-learn, `yfinance`.
- **Interface/exploration:** Streamlit, Jupyter, Matplotlib, seaborn, tqdm.
- **Storage:** local CSV datasets, PyTorch checkpoints, CSV reports, and PNG plots.

Dependencies and minimum versions are listed in [requirements.txt](requirements.txt).

## Architecture / How It Works

1. `StockDataLoader` loads or downloads a series, inspects it, and adds technical features.
2. `Preprocessor` fills missing values, clips outliers, scales features and the `Close` target separately, and builds sliding windows. Defaults are 60 historical observations and a one-step target, followed by a chronological 70%/15%/15% train/validation/test split. **Scaling currently happens before splitting; see limitations below.**
3. `Pipeline` constructs PyTorch data loaders and trains the selected models. Training batches are shuffled; validation and test batches retain their order.
4. `Trainer` uses Adam, MSE loss, gradient clipping, a validation-driven learning-rate scheduler, and early stopping, with automatic CUDA selection when available.
5. Predictions are inverse-scaled before evaluation. `ModelComparator` saves metrics and plots under `results/`; training saves checkpoints under `models/`.

The model implementations differ as follows:

- **RNN:** stacked `nn.RNN`, last-timestep readout, optional batch normalization, dropout, and a linear forecast head.
- **Transformer:** input projection, sinusoidal positions, PyTorch encoder layers with full attention within the historical window, and a last-position readout.
- **Causal Transformer:** learned positions, pre-normalization encoder layers, a cached causal attention mask, final layer normalization, and mean pooling.
- **Pretrained + fine-tuned Transformer:** trains the standard encoder on other tickers, copies its weights, then fine-tunes on the target series at one-fifth of the original learning rate. Each source ticker is scaled separately; the first 85% of its windows are pooled, with the final 10% of that pool used for pretraining validation.

## Technical Highlights

- Shared loading, preprocessing, training, and evaluation modules support both the GUI and programmatic experiments.
- A separate causal-attention variant changes positional encoding, normalization, masking, and pooling for architectural comparison.
- Cross-ticker pretraining and target-series fine-tuning are implemented as a two-stage training workflow.
- Evaluation includes MSE, RMSE, MAE, MAPE, R², directional agreement, and elapsed training time. Python, NumPy, and PyTorch seeds are set for repeatable experiments, although exact results depend on the environment and execution path.

## Setup

Use Python 3.10+ (3.11 is a reasonable starting point). From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`; in Windows Command Prompt, use `.venv\Scripts\activate.bat`.

Cached datasets allow the included experiment to run without downloading prices. Other symbols/date ranges require network access. Dependencies use minimum versions rather than a locked environment.

## Usage

Launch the interface from the repository root:

```bash
python -m streamlit run gui/app.py
```

Use **Data → Train Models → Predict / Comparison**. For the included transfer-learning setup, keep AAPL and the default date range, and select MSFT/GOOGL as pretraining sources. Extra source CSVs must already exist as `data/raw/SYMBOL_STARTDATE_ENDDATE.csv`. The UI trains models in the current session; its default prediction input is a historical test window.

For a programmatic run, save the following as a Python script in the repository root:

```python
from src.pipeline import Pipeline

pipeline = Pipeline(num_epochs=10)  # Short example; not the recorded benchmark.
pipeline.load_data("AAPL", "2020-01-01", "2024-01-01")
pipeline.preprocess_data()
pipeline.train_models(model_types=["rnn", "transformer", "transformer_causal"])
pipeline.evaluate_and_compare()
print(pipeline.predict("transformer", pipeline.X_test[-1:]))
```

To run the fourth configuration after preprocessing:

```python
pipeline.train_models(
    model_types=["transformer_pretrained_ft"],
    pretrain_tickers=["MSFT", "GOOGL"],
    primary_symbol="AAPL",
    primary_start_date="2020-01-01",
    primary_end_date="2024-01-01",
    pretrain_epochs=30,
)
pipeline.evaluate_and_compare()
```

Calling `train_models()` without model selections includes all four configurations and requires the primary-series metadata shown above. Runs overwrite matching checkpoint/report filenames by default; pass separate `models_dir` and `results_dir` values to `Pipeline` to retain experiments.

For the exploratory workflow, run `python -m jupyter notebook notebooks/main.ipynb` and execute cells in order.

## Project Structure

```text
data/raw/                      Cached AAPL, MSFT, and GOOGL price series
src/data_loader.py             Downloads, CSV loading, inspection, feature engineering
src/preprocessing.py           Cleaning, scaling, window creation, chronological splitting
src/models/                    RNN, Transformer, and causal Transformer implementations
src/training/                  Training loop, scheduling, early stopping, seed utilities
src/evaluation/                Metrics, model comparison, report generation
src/pipeline.py                End-to-end orchestration and transfer-learning workflow
gui/app.py                     Streamlit interface
notebooks/main.ipynb           Exploration and four-model experiment
models/                        Saved training checkpoints
results/                       Comparison CSV and plots
requirements.txt               Python dependencies
```

## Results / Metrics

The checked-in [comparison report](results/comparison_report.csv) records the following rounded values:

- **RNN:** RMSE **33.3755**, MAE **32.9991**, R² **-16.0660**, directional agreement **51.08%**.
- **Transformer:** RMSE **4.3103**, MAE **3.5816**, R² **0.7154**, directional agreement **53.24%**.
- **Causal Transformer:** RMSE **6.3040**, MAE **5.3028**, R² **0.3912**, directional agreement **45.32%**.
- **Pretrained + fine-tuned Transformer:** RMSE **3.0324**, MAE **2.4291**, R² **0.8591**, directional agreement **51.08%**.

These are recorded artifacts, not independently reproduced or leakage-free benchmark results. RMSE/MAE are evaluated on the inverse-scaled target. The CSV also stores MSE, MAPE, and training seconds, but does not identify the run's hardware, symbol, date range, or complete configuration. The notebook documents an AAPL experiment; it is not a provenance record for this exact CSV.

Directional agreement compares the signs of consecutive predicted values with consecutive actual values; it is not trading profitability. The metric flattens multi-step outputs, so its interpretation is especially limited for horizons greater than one.

## Limitations / Status

- **Experimental evaluation:** scalers and outlier thresholds use the full series before splitting, exposing preprocessing to validation/test data. The pretraining path does this too. Train-only preprocessing and time-aware validation are needed before claiming out-of-sample performance.
- **Early-stopping restoration:** `EarlyStopping` saves a shallow copy of `state_dict()`, whose tensors can change during training. Restoration is also only attempted when patience is exhausted. The evaluated weights are therefore not guaranteed to be from the best validation epoch.
- **Incomplete benchmarking:** no automated test suite, CI configuration, persistence/random-walk baseline, walk-forward evaluation, or repeated-seed uncertainty estimates are included.
- **Artifact reuse:** checkpoints are present, but the pipeline does not automatically save the fitted preprocessor alongside them. Reloading requires matching architecture, feature order, and scaling; the GUI has no checkpoint-loading workflow.
- This is a research/coursework application, with no demonstrated production deployment or trading backtest. Forecasts are not financial advice. No license file is included.

## Resume Evaluation Notes

- **Strongest technical aspects:** modular time-series pipeline, three neural architectures, cross-ticker transfer learning, and a Streamlit experiment interface.
- **Most impressive implementation:** the integrated pretraining/fine-tuning workflow and separate causal Transformer variant.
- **Weak/basic parts:** standard PyTorch encoder/RNN building blocks; preprocessing leakage, unreliable best-weight restoration, and limited benchmark/test coverage.
- **Verified metrics:** the committed CSV records a best RMSE of 3.0324 and R² of 0.8591 for the pretrained/fine-tuned model, subject to the limitations above; these are artifact values, not independently validated accuracy claims.
- **What is unique:** the combination of recurrent, full-attention, causal-attention, and cross-ticker transfer experiments in one local workflow; no novel algorithm is established.
- **What may be worth highlighting on a resume:** implementing and comparing sequence models, engineering financial time-series features, building shared training/evaluation tooling, and exposing experiments through Streamlit.
- **Claims that cannot be verified from the repo:** trading returns, real users, deployment scale, production reliability, statistically significant superiority, or generalization to unseen market regimes.
