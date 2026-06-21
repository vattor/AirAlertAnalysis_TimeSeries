import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Suppress warnings for cleaner output during tuning
warnings.filterwarnings('ignore')


def prepare_regional_data(df_raw, region_name, resolution='15min', lags=4):
    """
    Filters raw data for a specific region and resamples it to the target resolution.
    """
    region_df = df_raw[df_raw['region'] == region_name].copy()
    if region_df.empty:
        return None

    # Standardize start and end times
    min_time = region_df['started_at'].min().floor(resolution)
    max_time = region_df['finished_at'].max().ceil(resolution)

    # Create continuous timeline
    all_times = pd.date_range(start=min_time, end=max_time, freq=resolution)
    timeline_df = pd.DataFrame({'alert_active': 0}, index=all_times)

    # Mark active blocks
    for start, end in zip(region_df['started_at'], region_df['finished_at']):
        timeline_df.loc[start.floor(resolution):end.ceil(resolution), 'alert_active'] = 1

    # Feature Engineering (Lags and Time of Day)
    for i in range(1, lags + 1):
        timeline_df[f'lag_{i}'] = timeline_df['alert_active'].shift(i)

    timeline_df['hour_of_day'] = timeline_df.index.hour
    timeline_df.dropna(inplace=True)

    return timeline_df


def test_time_resolutions(local_path=r'..\data\volunteer_data_en.csv', lambda_penalty=1.5):
    """
    Tests different time gap resolutions across 3 stratified regions to find the universal optimum.
    """
    # 1. Define the Stratified Sample
    regions = {
        'High-Risk': 'Kharkivska oblast',
        'Medium-Risk': 'Kyiv City',
        'Low-Risk': 'Volynska oblast'
    }

    resolutions_to_test = ['30min']

    # Load raw data once to save memory
    print("Loading raw dataset...")
    df_raw = pd.read_csv(local_path)
    df_raw['started_at'] = pd.to_datetime(df_raw['started_at'])
    df_raw['finished_at'] = pd.to_datetime(df_raw['finished_at'])
    fill_time = df_raw['started_at'] + pd.Timedelta(minutes=30)
    df_raw['finished_at'] = df_raw['finished_at'].fillna(fill_time)

    results_tracker = []

    print(f"\nBeginning Universal Parameter Tuning (Penalty Lambda: {lambda_penalty})")
    print("-" * 65)

    # 2. Iterate over Resolutions
    for res in resolutions_to_test:
        print(f"Testing Resolution: {res}")
        region_aucs = []

        # 3. Iterate over Stratified Regions
        for risk_level, region_name in regions.items():
            data = prepare_regional_data(df_raw, region_name, resolution=res, lags=4)
            if data is None:
                continue

            features = [col for col in data.columns if col != 'alert_active']
            target = 'alert_active'
            X, y = data[features], data[target]

            # 4. Strict TimeSeries Validation (3 Splits to simulate forward progression)
            tscv = TimeSeriesSplit(n_splits=3)
            model = RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=15, random_state=42)

            fold_aucs = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # If the test set has no alerts (common in Low-Risk), AUC fails. Skip fold.
                if len(np.unique(y_test)) < 2:
                    continue

                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fold_aucs.append(roc_auc_score(y_test, y_pred_proba))

            # Average AUC for this specific region
            avg_region_auc = np.mean(fold_aucs) if fold_aucs else 0
            region_aucs.append(avg_region_auc)
            print(f"  -> {risk_level} ({region_name}): AUC = {avg_region_auc:.3f}")

        # 5. Calculate the Objective Function
        mean_auc = np.mean(region_aucs)
        variance_auc = np.var(region_aucs)
        objective_score = mean_auc - (lambda_penalty * variance_auc)

        results_tracker.append({
            'Resolution': res,
            'Mean AUC': mean_auc,
            'Variance': variance_auc,
            'Objective Score': objective_score
        })
        print(f"  [Result] Mean: {mean_auc:.3f} | Variance: {variance_auc:.4f} | Objective: {objective_score:.3f}\n")

    # 6. Final Recommendation
    results_df = pd.DataFrame(results_tracker)
    best_res = results_df.loc[results_df['Objective Score'].idxmax()]

    print("=== TUNING COMPLETE ===")
    print(results_df.to_string(index=False))
    print(f"\n🏆 RECOMMENDED RESOLUTION: {best_res['Resolution']} (Highest Objective Score)")


# Execute the test
if __name__ == "__main__":
    try:
        # Ensure your local path points to where your ETag script saves the file
        test_time_resolutions(local_path='volunteer_data_en.csv')
    except FileNotFoundError:
        print("Error: Could not find 'volunteer_data_en.csv'. Run the data ingestion script first.")