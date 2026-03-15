import pandas as pd

def add_lag_features(df, target_col='price', lags=[1, 2, 7]):
    """
    Add trailing lag features to time series dataframe dataframe.
    """
    # Important: Data must be sorted by date inside each region/group
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('region')[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col='price', windows=[7]):
    """
    Add rolling mean and standard deviation features.
    """
    df = df.copy()
    for w in windows:
        # Group by region to prevent cross-region rolling stats
        grouped = df.groupby('region')[target_col]
        df[f'rolling_mean_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'rolling_std_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df

def add_calendar_features(df, date_col='date'):
    """
    Extract calendar metadata from date column.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    return df

def encode_categorical_features(df):
    """
    One-hot encode categorical variables.
    """
    df = pd.get_dummies(df, columns=['region'], drop_first=False)
    # Convert booleans to int if needed by standard scikit models
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    return df

def generate_features(df):
    """
    Master pipeline to generate all features.
    Returns the dataframe with generated features and drops rows with NaNs caused by lagging.
    """
    # Sort ensures time-based operations like shift/rolling work correctly
    df = df.sort_values(by=['region', 'date']).reset_index(drop=True)
    
    df = add_lag_features(df, target_col='price', lags=[1, 2, 4])
    df = add_rolling_features(df, target_col='price', windows=[4, 8])
    df = add_calendar_features(df)
    df = encode_categorical_features(df)
    
    # Drop rows with NaNs that were produced by lagging
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv('data/synthetic_commodity_data.csv')
        df_featured = generate_features(df)
        out_path = 'data/featured_commodity_data.csv'
        df_featured.to_csv(out_path, index=False)
        print(f"Generated features and saved {len(df_featured)} rows to {out_path}.")
        print("Feature Columns generated:")
        print([col for col in df_featured.columns if col not in df.columns])
    except FileNotFoundError:
        print("Error: data/synthetic_commodity_data.csv not found. Please run data_generator.py first.")
