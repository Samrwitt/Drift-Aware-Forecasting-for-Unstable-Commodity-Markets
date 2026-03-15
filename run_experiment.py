import os
import pandas as pd
from src.data_generator import generate_synthetic_data
from src.features import generate_features
from src.models import run_forecasting_pipeline
from src.evaluation import calculate_metrics, plot_predictions, plot_mae_comparison

def main():
    print("="*50)
    print("Drift-Aware Forecasting for Unstable Commodity Markets")
    print("="*50)

    # 1. Setup Data Paths
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    raw_data_path = 'data/synthetic_commodity_data.csv'
    featured_data_path = 'data/featured_commodity_data.csv'
    
    # 2. Generate Data
    print("\n[1/4] Generating synthetic commodity data (Phase 1: Stable -> Phase 2: Shock -> Phase 3: Recovery)...")
    df_raw = generate_synthetic_data(n_steps=150, start_date='2020-01-01', random_state=42)
    df_raw.to_csv(raw_data_path, index=False)
    print(f"  Created raw dataset: {len(df_raw)} rows across {df_raw['region'].nunique()} regions.")

    # 3. Feature Engineering
    print("\n[2/4] Engineering features (Lags, Rolling Stats, Calendar, OHE)...")
    df_featured = generate_features(df_raw)
    df_featured.to_csv(featured_data_path, index=False)
    print(f"  Feature engineering complete. Total features ready for modeling: {len(df_featured.columns)}")
    
    features_to_use = [col for col in df_featured.columns if col not in ['date', 'region', 'commodity', 'price', 'shock_flag']]
    
    # 4. Modeling Pipeline
    models_to_run = ['naive', 'linear', 'random_forest'] # add 'xgboost' if desired
    strategies = ['static', 'adaptive']
    
    all_results = {}
    
    print("\n[3/4] Running walk-forward forecasting models...")
    for model_name in models_to_run:
        all_results[model_name] = {}
        print(f"  -> Training {model_name}...")
        for strategy in strategies:
            results_df = run_forecasting_pipeline(
                df=df_featured,
                features=features_to_use,
                target='price',
                model_name=model_name,
                strategy=strategy,
                train_window=40, # Initial stable block
                drift_threshold_ratio=1.5 # Retrain if rolling error is 1.5x training error
            )
            all_results[model_name][strategy] = results_df
            
            # Print brief summary
            overall_mae = results_df['error'].mean()
            print(f"     Strategy: {strategy:8} | Overall MAE: {overall_mae:.2f}")

    # 5. Evaluation and Verification
    print("\n[4/4] Generating evaluation metrics and plots...")
    
    for model_name in models_to_run:
        static_df = all_results[model_name]['static']
        adaptive_df = all_results[model_name]['adaptive']
        
        static_metrics = calculate_metrics(static_df)
        adaptive_metrics = calculate_metrics(adaptive_df)
        
        # Plot MAE comparison across regimes
        plot_mae_comparison(static_metrics, adaptive_metrics, model_name)
        
        # Plot actual vs predicted for a specific region (e.g., Addis Ababa)
        region_to_plot = 'Addis Ababa'
        plot_predictions(static_df, region_to_plot, model_name, 'static')
        plot_predictions(adaptive_df, region_to_plot, model_name, 'adaptive')
        
    print("\nExperiment complete! Check the 'outputs/' folder for regime comparisons and prediction plots.")
    print("="*50)

if __name__ == "__main__":
    main()
