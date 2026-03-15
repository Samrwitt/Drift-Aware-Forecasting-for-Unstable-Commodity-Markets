import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_metrics(results_df):
    """
    Calculate MAE and RMSE globally and per regime (stable vs shock).
    """
    metrics = {}
    mae = np.mean(results_df['error'])
    rmse = np.sqrt(np.mean(results_df['error']**2))
    
    metrics['overall'] = {'MAE': mae, 'RMSE': rmse}
    
    for shock_val, label in zip([0, 1], ['stable/recovery', 'shock']):
        subset = results_df[results_df['shock_flag'] == shock_val]
        if len(subset) > 0:
            metrics[label] = {
                'MAE': np.mean(subset['error']),
                'RMSE': np.sqrt(np.mean(subset['error']**2))
            }
    return metrics

def plot_predictions(results_df, region, model_name, strategy, output_dir='outputs'):
    """Plot actual vs predicted with uncertainty bounds and shock periods."""
    os.makedirs(output_dir, exist_ok=True)
    
    subset = results_df[results_df['region'] == region].copy()
    subset['date'] = subset['date'] # if not datetime, could convert
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset['date'], subset['actual'], label='Actual', color='black', linewidth=2)
    plt.plot(subset['date'], subset['predicted'], label=f'Predicted ({strategy})', color='blue')
    
    # Fill uncertainty
    plt.fill_between(subset['date'], subset['lower_bound'], subset['upper_bound'], 
                     color='blue', alpha=0.2, label='95% Confidence')
                     
    # Highlight shock period
    shock_dates = subset[subset['shock_flag'] == 1]['date']
    if not shock_dates.empty:
        plt.axvspan(shock_dates.min(), shock_dates.max(), color='red', alpha=0.1, label='Shock Period')
        
    plt.title(f"{model_name.capitalize()} Forecasting ({strategy.capitalize()}) - {region}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_{model_name}_{strategy}_{region.replace(' ', '_')}.png")
    plt.close()

def plot_mae_comparison(static_metrics, adaptive_metrics, model_name, output_dir='outputs'):
    """Compare MAE across regimes between static and adaptive."""
    os.makedirs(output_dir, exist_ok=True)
    
    categories = list(static_metrics.keys())
    static_mae = [static_metrics[cat]['MAE'] for cat in categories]
    adaptive_mae = [adaptive_metrics[cat]['MAE'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, static_mae, width, label='Static', color='lightcoral')
    ax.bar(x + width/2, adaptive_mae, width, label='Adaptive', color='mediumseagreen')
    
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title(f'MAE Comparison: Static vs Adaptive ({model_name.capitalize()})')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mae_comparison_{model_name}.png")
    plt.close()
