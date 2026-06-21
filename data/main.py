import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import requests
import os
import warnings

warnings.filterwarnings('ignore')

UKRAINE_ADJACENCY = {
    'Vinnytska oblast': ['Zhytomyrska oblast', 'Kyivska oblast', 'Cherkaska oblast', 'Kirovohradska oblast',
                         'Odeska oblast', 'Chernivetska oblast', 'Khmelnytska oblast'],
    'Volynska oblast': ['Rivnenska oblast', 'Lvivska oblast'],
    'Dnipropetrovska oblast': ['Poltavska oblast', 'Kharkivska oblast', 'Donetska oblast', 'Zaporizka oblast',
                               'Khersonska oblast', 'Mykolaivska oblast', 'Kirovohradska oblast'],
    'Donetska oblast': ['Luhanska oblast', 'Kharkivska oblast', 'Dnipropetrovska oblast', 'Zaporizka oblast'],
    'Zhytomyrska oblast': ['Rivnenska oblast', 'Khmelnytska oblast', 'Vinnytska oblast', 'Kyivska oblast'],
    'Zakarpattia oblast': ['Lvivska oblast', 'Ivano-Frankivska oblast'],
    'Zaporizka oblast': ['Donetska oblast', 'Dnipropetrovska oblast', 'Khersonska oblast'],
    'Ivano-Frankivska oblast': ['Zakarpattia oblast', 'Lvivska oblast', 'Ternopilska oblast', 'Chernivetska oblast'],
    'Kyivska oblast': ['Zhytomyrska oblast', 'Chernihivska oblast', 'Poltavska oblast', 'Cherkaska oblast',
                       'Vinnytska oblast'],
    'Kirovohradska oblast': ['Cherkaska oblast', 'Poltavska oblast', 'Dnipropetrovska oblast', 'Mykolaivska oblast',
                             'Odeska oblast', 'Vinnytska oblast'],
    'Luhanska oblast': ['Kharkivska oblast', 'Donetska oblast'],
    'Lvivska oblast': ['Volynska oblast', 'Rivnenska oblast', 'Ternopilska oblast', 'Ivano-Frankivska oblast',
                       'Zakarpattia oblast'],
    'Mykolaivska oblast': ['Odeska oblast', 'Kirovohradska oblast', 'Dnipropetrovska oblast', 'Khersonska oblast'],
    'Odeska oblast': ['Vinnytska oblast', 'Kirovohradska oblast', 'Mykolaivska oblast'],
    'Poltavska oblast': ['Chernihivska oblast', 'Sumska oblast', 'Kharkivska oblast', 'Dnipropetrovska oblast',
                         'Kirovohradska oblast', 'Cherkaska oblast', 'Kyivska oblast'],
    'Rivnenska oblast': ['Volynska oblast', 'Lvivska oblast', 'Ternopilska oblast', 'Khmelnytska oblast',
                         'Zhytomyrska oblast'],
    'Sumska oblast': ['Chernihivska oblast', 'Poltavska oblast', 'Kharkivska oblast'],
    'Ternopilska oblast': ['Lvivska oblast', 'Rivnenska oblast', 'Khmelnytska oblast', 'Chernivetska oblast',
                           'Ivano-Frankivska oblast'],
    'Kharkivska oblast': ['Luhanska oblast', 'Donetska oblast', 'Dnipropetrovska oblast', 'Poltavska oblast',
                          'Sumska oblast'],
    'Khersonska oblast': ['Zaporizka oblast', 'Dnipropetrovska oblast', 'Mykolaivska oblast'],
    'Khmelnytska oblast': ['Rivnenska oblast', 'Zhytomyrska oblast', 'Vinnytska oblast', 'Chernivetska oblast',
                           'Ternopilska oblast'],
    'Cherkaska oblast': ['Kyivska oblast', 'Poltavska oblast', 'Kirovohradska oblast', 'Vinnytska oblast'],
    'Chernivetska oblast': ['Ivano-Frankivska oblast', 'Ternopilska oblast', 'Khmelnytska oblast', 'Vinnytska oblast'],
    'Chernihivska oblast': ['Kyivska oblast', 'Sumska oblast', 'Poltavska oblast']
}




def download_raw_data(local_path='volunteer_data_en.csv'):
    """Handles ETag caching and downloads the raw dataset."""
    url = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
    etag_path = 'volunteer_data_en.etag'

    cached_etag = None
    if os.path.exists(etag_path):
        with open(etag_path, 'r') as f:
            cached_etag = f.read().strip()

    try:
        head_response = requests.head(url)
        current_etag = head_response.headers.get('ETag')

        if current_etag and cached_etag == current_etag:
            print("Dataset up to date. Using Cache.")
            return

        print("File changed or missing. Downloading...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            if current_etag:
                with open(etag_path, 'w') as f:
                    f.write(current_etag)
    except Exception as e:
        print(f"Network error: {e}. Attempting to use local cache.")


def build_unified_feature_matrix(df_raw, target_region, neighbor_list, target_lags=96, neighbor_lags=[2, 3, 4],
                                 freq='15min'):
    """
    User-Optimized Unified Function:
    1. Reverts to high-fidelity 15-minute tracking.
    2. Uses specific neighbor lags (e.g., [2, 3, 4]) to capture the 15-60 min travel window.
    3. Removes the "Current State" filter to retain the "All Clear" signal from neighbors.
    """
    print(f"Building unified matrix for target: {target_region} at {freq} resolution...")

    # 1. Establish the absolute timeline bounds
    target_data = df_raw[df_raw['region'] == target_region]
    min_time = target_data['started_at'].min().floor(freq)
    max_time = target_data['finished_at'].max().ceil(freq)
    master_index = pd.date_range(start=min_time, end=max_time, freq=freq)

    # 2. Initialize Master DataFrame
    master_df = pd.DataFrame(index=master_index)
    master_df['alert_active'] = 0

    # 3. Populate Target Status
    for start, end in zip(target_data['started_at'], target_data['finished_at']):
        master_df.loc[start.floor(freq):end.ceil(freq), 'alert_active'] = 1

    # 4. Generate Target Lags (Deep Memory)
    for i in range(1, target_lags + 1):
        master_df[f'lag_{i}'] = master_df['alert_active'].shift(i)

    # 5. Populate and Align Neighbor Lags (Targeted Window)
    new_neighbor_columns = {}
    for neighbor in neighbor_list:
        neighbor_data = df_raw[df_raw['region'] == neighbor]
        if len(neighbor_data) < 4:
            continue

        n_timeline = pd.Series(0, index=master_index)
        for start, end in zip(neighbor_data['started_at'], neighbor_data['finished_at']):
            try:
                n_timeline.loc[start.floor(freq):end.ceil(freq)] = 1
            except KeyError:
                pass

        safe_name = neighbor.replace(' ', '_').lower()

        # Iterating only through the specific lags the user requested (e.g., 2, 3, 4)
        for i in neighbor_lags:
            new_neighbor_columns[f'neighbor_{safe_name}_lag_{i}'] = n_timeline.shift(i)

    if new_neighbor_columns:
        master_df = pd.concat([master_df, pd.DataFrame(new_neighbor_columns)], axis=1)

    # 6. Temporal Features
    master_df['hour_of_day'] = master_df.index.hour
    master_df['day_of_week'] = master_df.index.dayofweek

    # 7. Safe Drop NA
    master_df.dropna(inplace=True)
    print(f"Matrix built successfully! Final shape: {master_df.shape}")

    return master_df


def train_and_evaluate(data, test_days=30, target_region='Unknown'):
    # FIXED FOR 15-MIN: 4 blocks/hr * 24 hrs = 96 blocks/day
    test_blocks = test_days * 96

    train_data = data.iloc[:-test_blocks]
    test_data = data.iloc[-test_blocks:]

    features = [col for col in data.columns if col != 'alert_active']
    target = 'alert_active'

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    print(f"\nTraining RandomForestClassifier on {len(features)} features...")
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print("\n=== Model Evaluation ===")
    print(f"Test Period: Last {test_days} days")
    print(f"ROC AUC Score: {auc_score:.3f}")

    custom_threshold = 0.3
    y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

    print(f"\nClassification Report (Custom Threshold: {int(custom_threshold * 100)}% Risk Tolerance):")
    print(classification_report(y_test, y_pred_custom))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
    plt.title(f'Receiver Operating Characteristic (ROC) for {target_region}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Feature Importance (Top 15)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    importances.head(15).plot(kind='bar', color='steelblue')
    plt.title(f'Top 15 Decision Features for {target_region}')
    plt.ylabel('Relative Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# --- Execution ---
LOCAL_FILE = 'volunteer_data_en.csv'
TARGET_REGION = 'Dnipropetrovska oblast'

try:
    AUTOMATED_NEIGHBORS = UKRAINE_ADJACENCY[TARGET_REGION]
    print(f"\n--- Initiating Analysis for {TARGET_REGION} ---")

    # Assuming download_raw_data() and UKRAINE_ADJACENCY are still in your script
    df_raw = pd.read_csv(LOCAL_FILE)
    df_raw['started_at'] = pd.to_datetime(df_raw['started_at'])
    df_raw['finished_at'] = pd.to_datetime(df_raw['finished_at'])
    df_raw['finished_at'] = df_raw['finished_at'].fillna(df_raw['started_at'] + pd.Timedelta(minutes=30))

    # 96 target lags = 24 hours of memory at 15-minute resolution
    final_dataset = build_unified_feature_matrix(
        df_raw=df_raw,
        target_region=TARGET_REGION,
        neighbor_list=AUTOMATED_NEIGHBORS,
        target_lags=96,
        neighbor_lags=[2, 3, 4],  # User's specific 15-60 minute window
        freq='15min'
    )

    train_and_evaluate(final_dataset, test_days=30, target_region=TARGET_REGION)

except Exception as e:
    print(f"An error occurred: {e}")



