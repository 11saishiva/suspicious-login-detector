from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime


class ImprovedSuspiciousLoginDetector:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.mandatory_features = [
            "login_timestamp", "ip_address", "latitude", "longitude",
            "is_blacklisted", "login_success"
        ]
        self.optional_features = [
            "user_id", "device_type", "browser", "is_off_hours", "pincode"
        ]

    def enhanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'login_timestamp' in df.columns:
            df['login_timestamp'] = pd.to_datetime(df['login_timestamp'])
            df['login_hour'] = df['login_timestamp'].dt.hour
            df['is_night'] = ((df['login_hour'] >= 0) & (df['login_hour'] <= 5)).astype(int)

        if 'latitude' in df.columns:
            df['geo_zone'] = pd.cut(df['latitude'], bins=3, labels=["north", "central", "south"])

        if 'ip_address' in df.columns:
            df['ip_subnet'] = df['ip_address'].apply(lambda x: '.'.join(x.split('.')[:2]) if isinstance(x, str) else "Unknown")

        return df

    def prepare_features_enhanced(self, df: pd.DataFrame, fit_encoders=False) -> Tuple[np.ndarray, pd.DataFrame]:
        df = df.copy()
        categorical_features = ['device_type', 'browser', 'geo_zone', 'ip_subnet']

        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(str).fillna("Unknown")
                if fit_encoders:
                    df[feature] = df[feature].astype(str).fillna("Unknown")
                    if fit_encoders:
                         unique_vals = df[feature].unique().tolist()
                         if "Unknown" not in unique_vals:
                              unique_vals.append("Unknown")
                              self.label_encoders[feature] = LabelEncoder()
                              self.label_encoders[feature].fit(unique_vals)
                              df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
                else:
                    known = set(self.label_encoders[feature].classes_)
                    df[feature] = df[feature].apply(lambda x: x if x in known else "Unknown")
                    df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
            else:
                df[f'{feature}_encoded'] = 0

        numeric_features = ['latitude', 'longitude', 'is_blacklisted', 'login_success', 'is_off_hours', 'pincode']
        for feature in numeric_features:
            if feature not in df.columns:
                df[feature] = 0.0

        all_features = [f"{f}_encoded" for f in categorical_features] + numeric_features
        X = df[all_features].fillna(0)
        if fit_encoders:
            self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        return X_scaled, df

    def create_balanced_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        risk_score = (
            3 * df.get('is_blacklisted', pd.Series([0]*len(df))) +
            2 * (1 - df.get('login_success', pd.Series([1]*len(df)))) +
            1 * df.get('is_off_hours', pd.Series([0]*len(df)))
        )
        threshold = np.percentile(risk_score, 65)
        labels = (risk_score >= threshold).astype(int)
        return labels, risk_score

    def train(self, df: pd.DataFrame):
        df = self.enhanced_feature_engineering(df)
        labels, _ = self.create_balanced_labels(df)
        X, _ = self.prepare_features_enhanced(df, fit_encoders=True)

        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, labels)
        self.model.fit(X_resampled, y_resampled)

    def predict_enhanced(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        df = df.copy()

        # Ensure mandatory features exist
        for col in self.mandatory_features:
            if col not in df.columns:
                raise ValueError(f"Missing mandatory feature: {col}")

        # Fill optional fields
        for col in self.optional_features:
            if col not in df.columns:
                if col in ['latitude', 'longitude', 'pincode']:
                    df[col] = 0.0
                elif col in ['is_off_hours']:
                    df[col] = 0
                else:
                    df[col] = "Unknown"

        df_features = self.enhanced_feature_engineering(df)
        X, _ = self.prepare_features_enhanced(df_features, fit_encoders=False)
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]
        return predictions, proba, df_features

    def save_model(self, model_path: str):
        joblib.dump(self, model_path)

    @staticmethod
    def load_model(filepath: str) -> 'ImprovedSuspiciousLoginDetector':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        print(f"Model loaded from: {filepath}")
        return joblib.load(filepath)

    def optimize_threshold(self, y_true, scores) -> float:
        best_thresh = 0.5
        best_f1 = 0
        for t in np.linspace(0.01, 0.5, 50):
            y_pred = (scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        return best_thresh


def run_improved_pipeline(data_path: str) -> Tuple[ImprovedSuspiciousLoginDetector, float]:
    df = pd.read_csv(data_path)
    df['login_timestamp'] = pd.to_datetime(df['login_timestamp'])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    detector = ImprovedSuspiciousLoginDetector()
    detector.train(train_df)

    test_labels, _ = detector.create_balanced_labels(test_df)
    _, scores, _ = detector.predict_enhanced(test_df)

    optimal_threshold = detector.optimize_threshold(test_labels, scores)
    model_filename = f"suspicious_login_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    detector.save_model(model_filename)
    print(f"Model saved to: {model_filename}")
    return detector, optimal_threshold
