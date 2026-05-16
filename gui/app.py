import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir) if current_dir.endswith('gui') else current_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_loader import StockDataLoader
from src.pipeline import Pipeline, SUPPORTED_MODELS

st.set_page_config(
    page_title="Stock Price Prediction - RNN vs Transformer",
    layout="wide",
)

st.title("Stock Price Prediction: RNN vs Transformer family")
st.markdown(
    "Train **Vanilla RNN**, **Transformer**, **Causal Transformer**, or "
    "**Transformer (pretrained + fine-tuned)** on one dataset, compare metrics, "
    "and run predictions. Pretraining pulls cached CSVs for extra symbols whose "
    "filenames match the date range you loaded."
)

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'data_load_meta' not in st.session_state:
    st.session_state.data_load_meta = None

tab_data, tab_train, tab_predict, tab_results = st.tabs(
    ["Data", "Train Models", "Predict", "Comparison"]
)

with tab_data:
    st.header("1. Load stock data")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock symbol", value="AAPL", help="e.g., AAPL, GOOGL, MSFT")
    with col2:
        start_date = st.date_input(
            "Start date",
            value=datetime(2020, 1, 1),
            max_value=datetime.now(),
        )
    with col3:
        end_date = st.date_input(
            "End date",
            value=datetime(2024, 1, 1),
            max_value=datetime.now(),
        )

    use_cached = st.checkbox(
        "Use cached CSV in data/raw if available",
        value=True,
    )

    if st.button("Load data"):
        with st.spinner("Loading data..."):
            try:
                loader = StockDataLoader(data_dir=os.path.join(parent_dir, "data", "raw"))
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                cached_path = os.path.join(
                    loader.data_dir, f"{symbol}_{start_str}_{end_str}.csv"
                )
                if use_cached and os.path.exists(cached_path):
                    data = loader.load_from_file(cached_path)
                    st.info(f"Loaded cached file: {cached_path}")
                else:
                    data = loader.fetch_data(
                        symbol=symbol,
                        start_date=start_str,
                        end_date=end_str,
                        save=True,
                    )

                data = loader.get_features(data)
                st.session_state.data = data
                st.session_state.data_load_meta = {
                    'primary_symbol': symbol.strip(),
                    'primary_start_date': start_str,
                    'primary_end_date': end_str,
                }

                st.success(f"Data loaded. Shape: {data.shape}")

                st.subheader("Preview")
                st.dataframe(data.head(10))

                st.subheader("Statistics")
                st.dataframe(data.describe())

                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                axes[0].plot(data['Date'], data['Close'])
                axes[0].set_title('Close price over time')
                axes[0].set_xlabel('Date'); axes[0].set_ylabel('Price')
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(data['Date'], data['Volume'])
                axes[1].set_title('Volume over time')
                axes[1].set_xlabel('Date'); axes[1].set_ylabel('Volume')
                axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as exc:
                st.error(f"Error loading data: {exc}")

with tab_train:
    st.header("2. Train models")

    if st.session_state.data is None:
        st.warning("Load a dataset first in the **Data** tab.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.slider("Sequence length", 10, 120, 60)
            prediction_horizon = st.slider("Prediction horizon", 1, 30, 1)
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
            learning_rate = st.number_input("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        with col2:
            hidden_size = st.slider(
                "Hidden / d_model", 32, 256, 64, step=8,
                help="Will be used as RNN hidden size and Transformer d_model.",
            )
            num_layers = st.slider("Number of layers", 1, 5, 2)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
            num_epochs = st.number_input("Max epochs", 10, 500, 50)

        use_batch_norm = st.checkbox("Use batch normalization", value=True)
        early_stopping = st.checkbox("Use early stopping", value=True)
        early_stopping_patience = (
            st.number_input("Early stopping patience", 5, 50, 10)
            if early_stopping else int(num_epochs)
        )

        pretrain_epochs = st.number_input(
            "Pretrain max epochs (pretrained + FT Transformer)",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
        )
        pretrain_ticker_choice = st.multiselect(
            "Pretrain ticker universe",
            options=["MSFT", "GOOGL", "AMZN", "META"],
            default=["MSFT", "GOOGL"],
            help="Used only when Transformer (pretrained + fine-tuned) is selected. "
            "Each ticker needs SYMBOL_START_END.csv in data/raw.",
        )

        model_labels = {
            "rnn": "Vanilla RNN",
            "transformer": "Transformer",
            "transformer_causal": "Causal Transformer (variant)",
            "transformer_pretrained_ft": "Transformer (pretrained + fine-tuned)",
        }
        selected = st.multiselect(
            "Models to train",
            options=list(SUPPORTED_MODELS),
            default=list(SUPPORTED_MODELS),
            format_func=lambda key: model_labels[key],
        )

        if st.button("Train selected models", type="primary"):
            if not selected:
                st.warning("Select at least one model.")
            else:
                progress = st.progress(0.0)
                status = st.empty()
                try:
                    pipeline = Pipeline(
                        sequence_length=sequence_length,
                        prediction_horizon=prediction_horizon,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        num_epochs=int(num_epochs),
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        use_batch_norm=use_batch_norm,
                        early_stopping_patience=int(early_stopping_patience),
                        data_dir=os.path.join(parent_dir, "data", "raw"),
                        models_dir=os.path.join(parent_dir, "models"),
                        results_dir=os.path.join(parent_dir, "results"),
                    )
                    pipeline.data = st.session_state.data

                    status.text("Preprocessing data...")
                    progress.progress(0.15)
                    pipeline.preprocess_data()
                    st.session_state.preprocessor = pipeline.preprocessor

                    meta = st.session_state.data_load_meta or {}
                    status.text(f"Training {len(selected)} model(s)...")
                    progress.progress(0.30)
                    pipeline.train_models(
                        selected,
                        pretrain_tickers=list(pretrain_ticker_choice),
                        pretrain_epochs=int(pretrain_epochs),
                        primary_symbol=meta.get('primary_symbol'),
                        primary_start_date=meta.get('primary_start_date'),
                        primary_end_date=meta.get('primary_end_date'),
                    )

                    status.text("Generating comparison report...")
                    progress.progress(0.90)
                    pipeline.evaluate_and_compare()

                    progress.progress(1.0)
                    status.text("Training complete.")
                    st.session_state.pipeline = pipeline
                    st.success(f"Trained {len(selected)} model(s) successfully.")

                    comparison_df = pipeline.comparator.compare_metrics()
                    st.subheader("Results summary")
                    st.dataframe(
                        comparison_df.style.format({
                            'mse': '{:.6f}',
                            'rmse': '{:.4f}',
                            'mae': '{:.4f}',
                            'mape': '{:.2f}',
                            'r2': '{:.4f}',
                            'directional_accuracy': '{:.2f}',
                            'training_time': '{:.2f}',
                        }),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Training error: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

with tab_predict:
    st.header("3. Make predictions")

    pipeline: Pipeline = st.session_state.pipeline
    if pipeline is None or not pipeline.trainers:
        st.warning("Train at least one model first in the **Train Models** tab.")
    else:
        trained_keys = list(pipeline.trainers.keys())
        model_labels = {
            "rnn": "Vanilla RNN",
            "transformer": "Transformer",
            "transformer_causal": "Causal Transformer (variant)",
            "transformer_pretrained_ft": "Transformer (pretrained + fine-tuned)",
        }
        model_choice = st.selectbox(
            "Trained model",
            options=trained_keys,
            format_func=lambda k: model_labels.get(k, k),
        )

        feature_cols = pipeline.preprocessor.feature_columns
        seq_len = pipeline.sequence_length
        num_features = len(feature_cols)

        st.markdown(
            f"Input requires the last **{seq_len} steps** for "
            f"**{num_features} features**: `{', '.join(feature_cols)}`."
        )

        use_last_test = st.checkbox(
            "Use last sequence from the test split (recommended)",
            value=True,
        )

        if use_last_test:
            X_scaled = pipeline.X_test[-1:].copy()
            preview_rows = min(10, seq_len)
            st.write(
                f"Preview of the last {preview_rows} scaled time-steps used as input:"
            )
            preview = X_scaled[0, -preview_rows:, :]
            st.dataframe(
                {feature_cols[i]: preview[:, i] for i in range(num_features)},
            )
            input_ready = True
        else:
            st.info(
                "Enter comma-separated raw values for each feature, "
                f"exactly {seq_len} per feature."
            )
            manual = {}
            for col in feature_cols:
                manual[col] = st.text_input(
                    f"{col} ({seq_len} values, comma-separated)",
                    value=",".join(["0.0"] * seq_len),
                )
            X_scaled = None
            input_ready = True
            if st.button("Validate manual input"):
                try:
                    rows = []
                    for col in feature_cols:
                        values = [float(x.strip()) for x in manual[col].split(",")]
                        if len(values) != seq_len:
                            raise ValueError(
                                f"{col} must have exactly {seq_len} values, got {len(values)}"
                            )
                        rows.append(values)
                    X_raw = np.array(rows).T.reshape(1, seq_len, num_features)
                    X_scaled = pipeline.preprocessor.scaler.transform(
                        X_raw.reshape(-1, num_features)
                    ).reshape(1, seq_len, num_features)
                    st.session_state['_manual_X_scaled'] = X_scaled
                    st.success("Manual input parsed and scaled successfully.")
                except Exception as exc:
                    st.error(f"Invalid manual input: {exc}")
                    input_ready = False
            X_scaled = st.session_state.get('_manual_X_scaled', None)

        if st.button("Predict", type="primary"):
            if X_scaled is None:
                st.error("No input prepared yet. Either tick the box above or validate manual input.")
            else:
                try:
                    prediction = pipeline.predict(
                        model_choice, X_scaled, return_original_scale=True
                    )
                    pred_value = float(prediction.flatten()[0])
                    st.success("Prediction complete.")
                    st.metric(
                        f"Predicted Close price ({model_labels.get(model_choice, model_choice)})",
                        f"${pred_value:,.2f}",
                    )

                    fig, ax = plt.subplots(figsize=(8, 4))
                    horizon = pipeline.prediction_horizon
                    ax.bar(range(horizon), prediction.flatten()[:horizon])
                    ax.set_xlabel('Step ahead')
                    ax.set_ylabel('Predicted price ($)')
                    ax.set_title(
                        f"Prediction ({model_labels.get(model_choice, model_choice)})"
                    )
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

with tab_results:
    st.header("4. Model comparison")

    pipeline = st.session_state.pipeline
    if pipeline is None or not pipeline.comparator.results:
        st.warning("Train at least one model first in the **Train Models** tab.")
    else:
        comparison_df = pipeline.comparator.compare_metrics()

        st.subheader("Best per metric")
        best_cols = st.columns(5)
        with best_cols[0]:
            best = comparison_df['rmse'].idxmin()
            st.metric("Best RMSE", best, f"{comparison_df.loc[best, 'rmse']:.4f}")
        with best_cols[1]:
            best = comparison_df['mae'].idxmin()
            st.metric("Best MAE", best, f"{comparison_df.loc[best, 'mae']:.4f}")
        with best_cols[2]:
            best = comparison_df['r2'].idxmax()
            st.metric("Best R²", best, f"{comparison_df.loc[best, 'r2']:.4f}")
        with best_cols[3]:
            best = comparison_df['training_time'].idxmin()
            st.metric("Fastest", best, f"{comparison_df.loc[best, 'training_time']:.1f}s")
        with best_cols[4]:
            best = comparison_df['directional_accuracy'].idxmax()
            st.metric(
                "Best directional acc.",
                best,
                f"{comparison_df.loc[best, 'directional_accuracy']:.2f}%",
            )

        st.subheader("Detailed comparison table")
        st.dataframe(
            comparison_df.style.format({
                'mse': '{:.6f}',
                'rmse': '{:.4f}',
                'mae': '{:.4f}',
                'mape': '{:.2f}',
                'r2': '{:.4f}',
                'directional_accuracy': '{:.2f}',
                'training_time': '{:.2f}',
            }),
            use_container_width=True,
        )

        st.subheader("Training and validation loss")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for model_name, result in pipeline.comparator.results.items():
            history = result.get('history') or {}
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label=model_name, alpha=0.85, linewidth=2)
            if 'val_loss' in history:
                axes[1].plot(history['val_loss'], label=model_name, alpha=0.85, linewidth=2)
        axes[0].set_title('Training loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_title('Validation loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Predictions vs actuals (test set)")
        num_samples = st.slider("Number of samples to plot", 50, 500, 200)
        n_models = len(pipeline.comparator.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 4.5 * n_models))
        if n_models == 1:
            axes = [axes]
        for idx, (model_name, result) in enumerate(pipeline.comparator.results.items()):
            y_true = result['y_true'].flatten()
            y_pred = result['y_pred'].flatten()
            indices = np.linspace(0, len(y_true) - 1, min(num_samples, len(y_true)), dtype=int)
            axes[idx].plot(indices, y_true[indices], label='Actual', linewidth=2)
            axes[idx].plot(indices, y_pred[indices], label='Predicted', linewidth=2, linestyle='--')
            axes[idx].set_title(f'{model_name} - Predictions vs Actual')
            axes[idx].set_xlabel('Sample index'); axes[idx].set_ylabel('Close price ($)')
            axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Training time")
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bars = len(comparison_df)
        bar_colors = [plt.cm.tab10(i % 10) for i in range(n_bars)]
        comparison_df['training_time'].plot(kind='bar', ax=ax, color=bar_colors)
        ax.set_ylabel('Seconds'); ax.set_xlabel('Model')
        ax.set_title('Training time per model'); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
