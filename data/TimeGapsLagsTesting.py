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



def optimize_resolution_and_lags(local_path=r'..\data\volunteer_data_en.csv', lambda_penalty=1.5):
    """
    Performs an expanded Grid Search over Resolution and Lag pairs (up to 48 hours).
    """
    # 1. Stratified Sample
    regions = {
        'High-Risk': 'Kharkivska oblast',
        'Medium-Risk': 'Kyiv City',
        'Low-Risk': 'Volynska oblast'
    }

    # 2. The EXPANDED Parameter Grid
    # We are now testing up to a 2-day memory capacity
    target_windows_hours = [12, 24, 48, 96]
    resolutions = ['30min']

    print("Loading raw dataset...")
    df_raw = pd.read_csv(local_path)
    df_raw['started_at'] = pd.to_datetime(df_raw['started_at'])
    df_raw['finished_at'] = pd.to_datetime(df_raw['finished_at'])
    fill_time = df_raw['started_at'] + pd.Timedelta(minutes=30)
    df_raw['finished_at'] = df_raw['finished_at'].fillna(fill_time)

    results_tracker = []

    print(f"\nBeginning Expanded Grid Search (Penalty Lambda: {lambda_penalty})")
    print("Warning: Testing 48-hour windows will take significantly longer to compute.")
    print("-" * 75)

    # 3. Nested Grid Search
    for res in resolutions:
        res_minutes = int(res.replace('min', ''))

        for window_hr in target_windows_hours:
            # Calculate required lags: (Hours * 60) / Resolution
            required_lags = int((window_hr * 60) / res_minutes)

            print(f"Testing Pair: {res} | Lags: {required_lags} (Total Lookback: {window_hr} hours)")
            region_aucs = []

            for risk_level, region_name in regions.items():
                # Prepare data with the dynamically calculated lags
                data = prepare_regional_data(df_raw, region_name, resolution=res, lags=required_lags)
                if data is None: continue

                features = [col for col in data.columns if col != 'alert_active']
                target = 'alert_active'
                X, y = data[features], data[target]

                tscv = TimeSeriesSplit(n_splits=3)
                model = RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=12, random_state=42)

                fold_aucs = []
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    if len(np.unique(y_test)) < 2: continue

                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fold_aucs.append(roc_auc_score(y_test, y_pred_proba))

                avg_region_auc = np.mean(fold_aucs) if fold_aucs else 0
                region_aucs.append(avg_region_auc)

            mean_auc = np.mean(region_aucs)
            variance_auc = np.var(region_aucs)
            objective_score = mean_auc - (lambda_penalty * variance_auc)

            results_tracker.append({
                'Resolution': res,
                'Lags': required_lags,
                'Lookback (Hr)': window_hr,
                'Mean AUC': mean_auc,
                'Variance': variance_auc,
                'Objective': objective_score
            })
            print(f"  [Result] Mean: {mean_auc:.3f} | Var: {variance_auc:.4f} | Objective: {objective_score:.3f}\n")

    # 4. Final Recommendation
    results_df = pd.DataFrame(results_tracker)
    best_pair = results_df.loc[results_df['Objective'].idxmax()]

    print("=== EXPANDED GRID SEARCH COMPLETE ===")
    print(results_df.sort_values(by='Objective', ascending=False).to_string(index=False))
    print(f"\n🏆 RECOMMENDED PAIR: {best_pair['Resolution']} with {best_pair['Lags']} Lags")
    print(f"This equates to an optimal lookback memory of {best_pair['Lookback (Hr)']} hours.")


if __name__ == "__main__":
    optimize_resolution_and_lags('volunteer_data_en.csv')
