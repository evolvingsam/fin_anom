import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import io
from datetime import datetime

# Step 1: Load historical dataset for context
data_str = """tran_dt,sender,receiver,amount,sender_prev_bal
2023-10-20,2301805351,1532644469,147.66,1295.2
2023-06-04,3703962220,2349736332,89.77,9479.55
2023-02-07,1529943053,3465910025,133.8,6677.19
2023-06-13,2555233206,7707731783,1251.02,4546.55
2023-08-03,1761232323,4532052055,55.01,6418.36
2023-08-04,2301805351,6580572805,118.95,1867.12
2023-05-29,2301805351,8293282371,194.16,1610.79
2023-03-02,2301805351,5888445457,941.74,9872.3
2023-08-06,2301805351,5782738976,243.02,3357.37
"""
df_historical = pd.read_csv(io.StringIO(data_str))
df_historical['tran_dt'] = pd.to_datetime(df_historical['tran_dt'])

# Step 2: Define the new transaction
new_transaction = {
    'tran_dt': '2023-08-07',  # Example new transaction
    'sender': 2301805351,
    'receiver': 9876543210,
    'amount': 500.00,
    'sender_prev_bal': 3000.00
}
df_new = pd.DataFrame([new_transaction])
df_new['tran_dt'] = pd.to_datetime(df_new['tran_dt'])

# Step 3: Feature engineering
def engineer_features(df, historical_df):
    df = df.copy()
    df['txn_hour'] = df['tran_dt'].dt.hour
    df['txn_day_of_week'] = df['tran_dt'].dt.dayofweek
    df['sender_post_bal'] = df['sender_prev_bal'] - df['amount']
    df['amount_to_balance_ratio'] = df['amount'] / (df['sender_prev_bal'] + 1e-6)
    df['date'] = df['tran_dt'].dt.date
    
    # Compute daily transaction count and unique receivers from historical data
    sender = df['sender'].iloc[0]
    date = df['date'].iloc[0]
    historical_sender = historical_df[historical_df['sender'] == sender]
    df['daily_txn_count'] = len(historical_sender[historical_sender['tran_dt'].dt.date == date]) + 1  # Include new transaction
    df['unique_receivers'] = historical_sender['receiver'].nunique() + 1  # Include new receiver
    
    # Impute categorical features
    df['txn_type'] = 'transfer'
    df['sender_acc_type'] = 'personal'
    df['txn_location'] = 'NYC'
    return df

df_new = engineer_features(df_new, df_historical)

# Step 4: Preprocess data
numerical_cols = ['amount', 'sender_prev_bal', 'sender_post_bal', 'txn_hour', 'txn_day_of_week', 
                 'amount_to_balance_ratio', 'daily_txn_count', 'unique_receivers']
categorical_cols = ['txn_type', 'sender_acc_type', 'txn_location']

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_new_categorical = encoder.fit_transform(df_new[categorical_cols])
categorical_encoded_cols = encoder.get_feature_names_out(categorical_cols)
X_new_categorical_df = pd.DataFrame(X_new_categorical, columns=categorical_encoded_cols)

# Combine features
X_new = pd.concat([df_new[numerical_cols].reset_index(drop=True), X_new_categorical_df], axis=1)

# Standardize features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Apply PCA
pca = PCA(n_components=5)
X_new_pca = pca.fit_transform(X_new_scaled)

# Step 5: Supervised Fraud Detection (using pre-trained Random Forest)
# Train Random Forest on hypothetical data (simplified for demo)
hypo_data = pd.DataFrame({
    'tran_dt': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(1000)],
    'sender_id': np.random.randint(1000000000, 9999999999, 1000),
    'receiver_id': np.random.randint(1000000000, 9999999999, 1000),
    'amount': np.random.lognormal(mean=5, sigma=1.5, size=1000).round(2),
    'txn_type': np.random.choice(['purchase', 'transfer', 'withdrawal'], 1000),
    'sender_acc_type': np.random.choice(['personal', 'business'], 1000),
    'sender_prev_bal': np.random.lognormal(mean=8, sigma=1.5, size=1000).round(2),
    'txn_location': np.random.choice(['NYC', 'LAX', 'CHI', 'LON', 'TOK'], 1000),
    'is_fraud': np.random.choice([0, 1], 1000, p=[0.98, 0.02])
})
hypo_data['sender_post_bal'] = hypo_data['sender_prev_bal'] - hypo_data['amount']
hypo_data = engineer_features(hypo_data, hypo_data)

X_hypo = pd.concat([
    hypo_data[numerical_cols],
    pd.DataFrame(encoder.fit_transform(hypo_data[categorical_cols]), columns=categorical_encoded_cols)
], axis=1)
X_hypo_scaled = scaler.fit_transform(X_hypo)
X_hypo_pca = pca.fit_transform(X_hypo_scaled)
y_hypo = hypo_data['is_fraud']

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_hypo_pca, y_hypo)

# Predict fraud
fraud_prob = rf_model.predict_proba(X_new_pca)[0, 1]
is_fraud_pred = rf_model.predict(X_new_pca)[0]
print("\nSupervised Fraud Detection Result:")
print(f"Transaction: {new_transaction}")
print(f"Fraud Probability: {fraud_prob:.2%}")
print(f"Prediction: {'Fraud' if is_fraud_pred == 1 else 'Not Fraud'}")

# Step 6: Unsupervised Fraud Detection (Isolation Forest)
# Train on historical data for sender
sender_historical = df_historical[df_historical['sender'] == new_transaction['sender']].copy()
sender_historical = engineer_features(sender_historical, df_historical)
X_sender = pd.concat([
    sender_historical[numerical_cols],
    pd.DataFrame(encoder.transform(sender_historical[categorical_cols]), columns=categorical_encoded_cols)
], axis=1)
X_sender_scaled = scaler.fit_transform(X_sender)
X_sender_pca = pca.fit_transform(X_sender_scaled)

# Combine new transaction with historical for context
X_combined_pca = np.vstack([X_sender_pca, X_new_pca])
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_combined_pca)
is_anomaly = anomaly_labels[-1]  # Last row is the new transaction
print("\nUnsupervised Fraud Detection Result:")
print(f"Transaction: {new_transaction}")
print(f"Prediction: {'Fraud' if is_anomaly == -1 else 'Not Fraud'}")

# Step 7: Save result
df_new['predicted_fraud'] = is_fraud_pred
df_new['fraud_probability'] = fraud_prob
df_new['is_anomaly'] = is_anomaly == -1
df_new.to_csv('new_transaction_result.csv', index=False)
print("Result saved to 'new_transaction_result.csv'")