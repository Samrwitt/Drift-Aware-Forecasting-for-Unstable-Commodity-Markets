import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_steps=150, start_date='2020-01-01', random_state=42):
    """
    Generate synthetic commodity market data for two regions with three phases:
    1. Stable phase (0 - 50)
    2. Shock phase (50 - 100)
    3. Recovery phase (100 - 150)
    """
    np.random.seed(random_state)
    
    dates = pd.date_range(start=start_date, periods=n_steps, freq='W')
    regions = ['Addis Ababa', 'Adama']
    
    data = []
    
    for region in regions:
        # Base price differences between regions
        base_price = 3000 if region == 'Addis Ababa' else 2800
        
        # Phases definitions
        stable_idx = 50
        shock_idx = 100
        
        prices = []
        exchange_rates = []
        rainfall_proxies = []
        holiday_flags = []
        shock_flags = []
        
        current_price = base_price
        current_exchange_rate = 50.0 # Base ETB/USD
        
        for i in range(n_steps):
            # Phase 1: Stable
            if i < stable_idx:
                shock_flag = 0
                price_volatility = 50
                exchange_rate_drift = 0.05
                rainfall = max(0, np.sin(i / 5.0) * 50 + 50 + np.random.normal(0, 10))
                
            # Phase 2: Shock
            elif i < shock_idx:
                shock_flag = 1
                price_volatility = 200
                exchange_rate_drift = 0.5 # Rapid devaluation
                rainfall = max(0, np.sin(i / 5.0) * 30 + 30 + np.random.normal(0, 15)) # Maybe a drought
                
                # Big shock upward drift
                current_price += np.random.normal(100, 50) 
                
            # Phase 3: Recovery
            else:
                shock_flag = 0
                price_volatility = 80
                exchange_rate_drift = 0.02 # Stabilized but at new high
                rainfall = max(0, np.sin(i / 5.0) * 50 + 50 + np.random.normal(0, 10))
                
                # Slow downward/stabilizing drift
                current_price -= np.random.normal(20, 10)
                # Keep floor
                current_price = max(base_price * 1.5, current_price)
                
            # Random holiday spikes (~5% chance)
            is_holiday = int(np.random.random() < 0.05)
            holiday_bump = np.random.normal(300, 100) if is_holiday else 0
            
            # Formulate current timestep
            current_exchange_rate += exchange_rate_drift + np.random.normal(0, 0.1)
            final_price = current_price + np.random.normal(0, price_volatility) + holiday_bump
            
            # Region specific noise
            if region == 'Adama':
                final_price *= 0.95 # slightly cheaper in Adama
            
            prices.append(max(100, final_price)) # No negative prices
            exchange_rates.append(current_exchange_rate)
            rainfall_proxies.append(rainfall)
            holiday_flags.append(is_holiday)
            shock_flags.append(shock_flag)
            
        df_region = pd.DataFrame({
            'date': dates,
            'region': region,
            'commodity': 'Teff',
            'price': prices,
            'exchange_rate': exchange_rates,
            'rainfall_proxy': rainfall_proxies,
            'holiday_flag': holiday_flags,
            'shock_flag': shock_flags
        })
        data.append(df_region)
        
    df_final = pd.concat(data, ignore_index=True)
    df_final = df_final.sort_values(by=['date', 'region']).reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(n_steps=150)
    
    out_path = 'data/synthetic_commodity_data.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with {len(df)} rows to {out_path}")
    print("\nSample:")
    print(df.head())
    print("\nSummary Stats by Phase:")
    print(df.groupby('shock_flag')['price'].describe())
