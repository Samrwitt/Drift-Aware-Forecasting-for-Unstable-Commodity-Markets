import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class NaiveModel:
    """Predicts the exact previous value."""
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        # Assumes price_lag_1 is the first column or explicitly provided
        # We will extract price_lag_1 from X
        if 'price_lag_1' in X.columns:
            return X['price_lag_1'].values
        return np.zeros(len(X))

def get_model(model_name):
    """Factory to get the requested model."""
    if model_name == 'naive':
        return NaiveModel()
    elif model_name == 'linear':
        return LinearRegression()
    elif model_name == 'random_forest':
        return RandomForestRegressor(n_estimators=50, random_state=42)
    elif model_name == 'xgboost':
        return xgb.XGBRegressor(n_estimators=50, random_state=42, objective='reg:squarederror')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def calculate_uncertainty(recent_errors, scale=1.96):
    """
    Calculate confidence intervals based on recent prediction errors.
    Returns margin, and categorical confidence level.
    """
    if len(recent_errors) < 2:
        return 0, 'low'
    
    std_error = np.std(recent_errors)
    margin = scale * std_error
    
    # Simple heuristic for confidence label
    if std_error < 50:
        conf = 'high'
    elif std_error < 150:
        conf = 'medium'
    else:
        conf = 'low'
        
    return margin, conf

def run_forecasting_pipeline(df, features, target='price', model_name='linear', 
                            strategy='static', train_window=30, drift_threshold_ratio=1.5):
    """
    Run walk-forward validation for a given strategy.
    
    strategy: 'static' or 'adaptive'
    """
    df = df.copy()
    
    # We will iterate row by row over the testing period.
    # To maintain temporal strictness, we assume df is sorted by date for a specific region.
    # Group by region to prevent cross-region data leakage during walk-forward
    
    results = []
    
    for region in df['region'].unique():
        df_region = df[df['region'] == region].reset_index(drop=True)
        
        # Initial training pool length
        current_train_end = train_window
        if len(df_region) <= current_train_end:
            continue
            
        model = get_model(model_name)
        
        # Initial training
        X_train = df_region.loc[:current_train_end-1, features]
        y_train = df_region.loc[:current_train_end-1, target]
        model.fit(X_train, y_train)
        
        baseline_mae = np.mean(np.abs(model.predict(X_train) - y_train))
        # Add a floor to baseline MAE to prevent div-by-zero or extreme sensitivity
        baseline_mae = max(baseline_mae, 10.0) 
        
        recent_errors = []
        rolling_mae_window = 5
        
        for i in range(current_train_end, len(df_region)):
            row = df_region.iloc[i:i+1]
            X_test = row[features]
            y_true = row[target].values[0]
            
            # Predict
            y_pred = model.predict(X_test)[0]
            
            # Error and Uncertainty Estimation
            error = np.abs(y_pred - y_true)
            
            margin, conf = calculate_uncertainty(recent_errors, scale=1.96)
            
            results.append({
                'date': row['date'].values[0],
                'region': region,
                'actual': y_true,
                'predicted': y_pred,
                'lower_bound': y_pred - margin,
                'upper_bound': y_pred + margin,
                'confidence': conf,
                'error': error,
                'shock_flag': row['shock_flag'].values[0]
            })
            
            # Update error history
            recent_errors.append(error)
            if len(recent_errors) > rolling_mae_window:
                recent_errors.pop(0)
                
            # Drift Detection
            current_rolling_mae = np.mean(recent_errors)
            
            if strategy == 'adaptive' and current_rolling_mae > baseline_mae * drift_threshold_ratio:
                # Drift detected! Retrain on recent window
                # print(f"[{region}] Drift detected at {row['date'].values[0]}! Retraining...")
                retrain_start = max(0, i - train_window)
                X_retrain = df_region.loc[retrain_start:i, features]
                y_retrain = df_region.loc[retrain_start:i, target]
                model.fit(X_retrain, y_retrain)
                
                # Reset baseline and error history after adaptation
                baseline_mae = max(10.0, np.mean(np.abs(model.predict(X_retrain) - y_retrain)))
                recent_errors = []
                
    return pd.DataFrame(results)
