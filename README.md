# Stock Price Forecasting — RNN & Transformer (PyTorch)

End-to-end deep learning project for **sequence modeling on stock OHLCV data**: fetch or load prices, engineer features, train sequence models, evaluate metrics, and compare architectures.

## What’s included

- **Data**: `yfinance` downloads with optional CSV cache (`src/data_loader.py`); basic inspection and technical features (returns, moving averages, volatility, volume ratios).
- **Models**:
  - **Vanilla RNN** — baseline recurrent sequence model.
  - **Transformer** — encoder-style model with sinusoidal positional encoding.
  - **Causal Transformer** — learnable positions, pre-norm blocks, causal masking, mean pooling (`src/models/transformer_variant.py`).
- **Pipeline**: `src/pipeline.py` ties loading, preprocessing, training, saving checkpoints, and comparison reports under `results/`.
- **Notebook**: `notebooks/main.ipynb` for exploratory workflows.
- **GUI**: Streamlit app for loading data, training, predicting, and comparing models (`gui/app.py`).

## Requirements

- Python 3.10+ recommended  
- Dependencies are listed in `requirements.txt` (PyTorch, pandas, scikit-learn, matplotlib, streamlit, yfinance, Jupyter, etc.)

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(On macOS/Linux use `source .venv/bin/activate`.)

## Run the Streamlit UI

From the repository root (so `src` resolves correctly):

```bash
streamlit run gui/app.py
```

## Use the pipeline in code

Instantiate `Pipeline` from `src.pipeline`, load data via `StockDataLoader`, then call the pipeline methods for training and comparison. Supported model keys are: `rnn`, `transformer`, and `transformer_causal` (see `SUPPORTED_MODELS` in `src/pipeline.py`).

Trained weights are written under `models/`; reports and plots under `results/` when you run the full evaluation flow.

## Note on forecasting

Models here are for **research and coursework**: markets are non-stationary and noisy. Do not treat outputs as financial advice.

## License

Add a `LICENSE` file if you plan to open-source the repo publicly.
