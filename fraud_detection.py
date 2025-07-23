import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import io
from datetime import datetime, timedelta

# Load historical dataset for context

df_historical = pd.read_csv("transactions_patterned.csv")
df_historical['tran_dt'] = pd.to_datetime(df_historical['tran_dt'])

new_transaction = {
    'tran_dt': '2023-08-07',  
    'sender': 2301805351,
    'receiver': 9876543210,
    'amount': 500.00,
    'sender_prev_bal': 3000.00
}
df_new = pd.DataFrame([new_transaction])
df_new['tran_dt'] = pd.to_datetime(df_new['tran_dt'])


def engineer_features(df, historical_df):
    df = df.copy()
    df['txn_hour'] = df['tran_dt'].dt.hour
    df['txn_day_of_week'] = df['tran_dt'].dt.dayofweek
    df['sender_post_bal'] = df['sender_prev_bal'] - df['amount']
    df['amount_to_balance_ratio'] = df['amount'] / (df['sender_prev_bal'] + 1e-6)
    df['date'] = df['tran_dt'].dt.date
    
    
    sender = df['sender'].iloc[0]
    date = df['date'].iloc[0]
    historical_sender = historical_df[historical_df['sender'] == sender]
    existing_count = len(historical_sender[historical_sender['tran_dt'].dt.date == date])
    already_present = (
        (historical_sender['tran_dt'].dt.date == date) &
        (historical_sender['receiver'] == df['receiver'].iloc[0]) &
        (np.isclose(historical_sender['amount'], df['amount'].iloc[0]))
    ).any()
    df['daily_txn_count'] = existing_count + (0 if already_present else 1)
    new_receiver = df['receiver'].iloc[0]
    unique_receivers = historical_sender['receiver'].nunique()
    if new_receiver not in historical_sender['receiver'].values:
        unique_receivers += 1
    df['unique_receivers'] = unique_receivers
    
    df['txn_type'] = 'transfer'
    df['sender_acc_type'] = 'personal'
    df['txn_location'] = 'NYC'
    return df

df_new = engineer_features(df_new, df_historical)

# Step 4: Preprocess data
numerical_cols = ['amount', 'sender_prev_bal', 'sender_post_bal', 'txn_hour', 'txn_day_of_week', 
                 'amount_to_balance_ratio', 'daily_txn_count', 'unique_receivers']
categorical_cols = ['txn_type', 'sender_acc_type', 'txn_location']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()
pca = PCA(n_components=5)

# Supervised Fraud Detection (using pre-trained Random Forest)
np.random.seed(42)
hypo_data = pd.DataFrame({
    'tran_dt': [datetime(2023, 1, 1) + timedelta(days=i % 365) for i in range(10000)],
    'sender': np.random.randint(1000000000, 9999999999, 10000),
    'receiver': np.random.randint(1000000000, 9999999999, 10000),
    'amount': np.random.lognormal(mean=5, sigma=1.5, size=10000).round(2),
    'txn_type': np.random.choice(['purchase', 'transfer', 'withdrawal'], 10000),
    'sender_acc_type': np.random.choice(['personal', 'business'], 10000),
    'sender_prev_bal': np.random.lognormal(mean=8, sigma=1.5, size=10000).round(2),
    'txn_location': np.random.choice(['NYC', 'LAX', 'CHI', 'LON', 'TOK'], 10000)
})

# Add realistic fraud labels (e.g., high amounts at odd hours)
hypo_data['is_fraud'] = 0
hypo_data.loc[
    (hypo_data['amount'] > hypo_data['amount'].quantile(0.95)) & 
    (hypo_data['tran_dt'].dt.hour.isin([0, 1, 2, 3, 22, 23])), 
    'is_fraud'] = 1
hypo_data['sender_post_bal'] = hypo_data['sender_prev_bal'] - hypo_data['amount']
hypo_data = engineer_features(hypo_data, hypo_data)

X_hypo_categorical = encoder.fit_transform(hypo_data[categorical_cols])
categorical_encoded_cols = encoder.get_feature_names_out(categorical_cols)
X_hypo_categorical_df = pd.DataFrame(X_hypo_categorical, columns=categorical_encoded_cols)
X_hypo = pd.concat([
    hypo_data[numerical_cols].reset_index(drop=True),
    X_hypo_categorical_df
], axis=1)
X_hypo_scaled = scaler.fit_transform(X_hypo)
X_hypo_pca = pca.fit_transform(X_hypo_scaled)
y_hypo = hypo_data['is_fraud']

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_hypo_pca, y_hypo)

X_new_categorical = encoder.transform(df_new[categorical_cols])
X_new_categorical_df = pd.DataFrame(X_new_categorical, columns=categorical_encoded_cols)
X_new = pd.concat([df_new[numerical_cols].reset_index(drop=True), X_new_categorical_df], axis=1)
X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)

fraud_prob = rf_model.predict_proba(X_new_pca)[0, 1]
is_fraud_pred = rf_model.predict(X_new_pca)[0]
print("\nSupervised Fraud Detection Result:")
print(f"Transaction: {new_transaction}")
print(f"Fraud Probability: {fraud_prob:.2%}")
print(f"Prediction: {'Fraud' if is_fraud_pred == 1 else 'Not Fraud'}")


sender_historical = df_historical[df_historical['sender'] == new_transaction['sender']].copy()
sender_historical = engineer_features(sender_historical, df_historical)
X_sender_categorical = encoder.transform(sender_historical[categorical_cols])
X_sender_categorical_df = pd.DataFrame(X_sender_categorical, columns=categorical_encoded_cols)
X_sender = pd.concat([
    sender_historical[numerical_cols].reset_index(drop=True),
    X_sender_categorical_df
], axis=1)
X_sender_scaled = scaler.transform(X_sender)
X_sender_pca = pca.transform(X_sender_scaled)

X_combined_pca = np.vstack([X_sender_pca, X_new_pca])
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_combined_pca)
is_anomaly = anomaly_labels[-1] 
print("\nUnsupervised Fraud Detection Result:")
print(f"Transaction: {new_transaction}")
print(f"Prediction: {'Fraud' if is_anomaly == -1 else 'Not Fraud'}")

df_new['predicted_fraud'] = is_fraud_pred
df_new['fraud_probability'] = fraud_prob
df_new['is_anomaly'] = is_anomaly == -1
df_new.to_csv('new_transaction_result.csv', index=False)
print("Result saved to 'new_transaction_result.csv'")