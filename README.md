# Drift-Aware Forecasting for Unstable Commodity Markets

This prototype implements a forecasting pipeline specifically designed for highly volatile commodity markets, inspired by Ethiopian economic conditions. Traditional time series forecasting models typically assume underlying market stability. However, in emerging markets facing structural shocks (such as currency devaluation, drought, or socio-political events), these assumptions fail.

This project demonstrates the impact of **structural drift** on static forecasting models and provides an **adaptive modeling strategy** that automatically detects volatility and retrains the model in real-time.

## Key Features
1. **Semi-Synthetic Environment Generator**: Creates a multi-region commodity price dataset encompassing three distinct phases: Stable, Shock, and Recovery.
2. **Feature Engineering**: Implements robust time series feature extraction including price lags, rolling statistics, and calendar metadata.
3. **Drift Detection**: Uses a walk-forward evaluation protocol with a rolling Mean Absolute Error (MAE) monitor to detect when the market dynamics break from the initial training distribution.
4. **Adaptive Retraining**: When drift is triggered, the adaptive model flushes obsolete historical data and retrains on the most recent behavior block.
5. **Uncertainty Estimation**: Rather than predicting exact point values, the system outputs prediction intervals representing confidence bands based on recent residual magnitudes.

## Project Structure
```text
Drift-Aware-Forecasting-for-Unstable-Commodity-Markets/
│
├── data/                  # Contains generated synthetic datasets
├── outputs/               # Auto-generated plots and regime evaluations
├── src/
│   ├── data_generator.py  # Environment synthesis across phases
│   ├── features.py        # Lag and rolling feature generation
│   ├── models.py          # Prediction wrappers and drift mechanics
│   └── evaluation.py      # Plotting and numerical metrics
│
├── run_experiment.py      # End-to-end execution script
├── requirements.txt       # Project dependencies
└── README.md              # Technical documentation
```

## Running the Experiment

To run the experiment, ensure you have Python installed, then set up the environment and execute the run script:

```bash
# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the end-to-end demonstration
python run_experiment.py
```

## Static vs. Adaptive Comparison

The experiment compares two strategies:
- **Static Model**: Trained on initial stable data. Once the *Shock* phase occurs, error margins explode and the model fails to recover or map the new market plateau.
- **Adaptive Model**: Tracks recent rolling MAE compared to the baseline MAE from training. When the ratio crosses a threshold (e.g., 1.5x), the model throws away early stable data and retrains exclusively on recent shock behaviors, substantially minimizing forecasting error during periods of rapid change.

## Conclusion and Limitations
**Prototype Scope:** This is a research prototype targeting the demonstration of drift detection logic within time series. 
**Limitations:** 
- The synthetic dataset is intentionally exaggerated to clearly delineate stable vs shock regimes.
- The simplistic error-based drift detection might trigger frequently if the data is overly noisy without structural breaks. 
- Real-world integration requires actual macro-economic data (trusted exchange rates, rain-gauge data, etc.).
