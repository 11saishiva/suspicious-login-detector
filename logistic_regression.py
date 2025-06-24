# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, RobustScaler
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import SMOTE
# from datetime import datetime, timedelta
# import warnings
# import joblib
# import os
# warnings.filterwarnings('ignore')

# class ImprovedSuspiciousLoginDetector:
#     def __init__(self, contamination=0.1):
#         self.contamination = contamination
#         self.method = 'logistic_regression'

#         self.model = LogisticRegression(
#             random_state=42,
#             class_weight='balanced',
#             max_iter=1000
#         )
        
#         self.label_encoders = {}
#         self.scaler = RobustScaler()  
#         self.feature_names = []
#         self.feature_importance = {}
        
#     def enhanced_feature_engineering(self, df):
        
#         features_df = df.copy()
        
        
#         if 'login_timestamp' in features_df.columns:
#             features_df['hour'] = features_df['login_timestamp'].dt.hour
#             features_df['day_of_week'] = features_df['login_timestamp'].dt.dayofweek
#             features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
#             features_df['is_night'] = ((features_df['hour'] >= 21) | (features_df['hour'] <= 6)).astype(int)
            
            
#             features_df['is_business_hours'] = ((features_df['hour'] >= 6) & (features_df['hour'] <= 19)).astype(int)
#             features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
#             features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
#             features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
#             features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        
#         if 'latitude' in features_df.columns and 'longitude' in features_df.columns:
#             user_locations = features_df.groupby('user_id')[['latitude', 'longitude']].agg(['mean', 'std']).reset_index()
#             user_locations.columns = ['user_id', 'lat_mean', 'lon_mean', 'lat_std', 'lon_std']
#             features_df = features_df.merge(user_locations, on='user_id', how='left')
            
            
#             features_df['distance_from_usual'] = np.sqrt(
#                 (features_df['latitude'] - features_df['lat_mean'])**2 + 
#                 (features_df['longitude'] - features_df['lon_mean'])**2
#             ) * 111  
            
            
#             features_df['location_consistency'] = 1 / (1 + features_df['lat_std'].fillna(0) + features_df['lon_std'].fillna(0))
#             features_df['is_new_location'] = (features_df['distance_from_usual'] > 50).astype(int)  # >50km threshold
        
        
#         if 'ip_address' in features_df.columns:
#             ip_counts = features_df['ip_address'].value_counts()
#             features_df['ip_frequency'] = features_df['ip_address'].map(ip_counts)
#             features_df['ip_rarity'] = 1 / features_df['ip_frequency']
            
#             user_ip_counts = features_df.groupby('user_id')['ip_address'].nunique()
#             features_df['user_ip_diversity'] = features_df['user_id'].map(user_ip_counts)
#             features_df['ip_subnet'] = features_df['ip_address'].apply(
#                 lambda x: '.'.join(str(x).split('.')[:3]) if pd.notna(x) and '.' in str(x) else 'Unknown'
#             )
#             subnet_counts = features_df['ip_subnet'].value_counts()
#             features_df['subnet_frequency'] = features_df['ip_subnet'].map(subnet_counts)
        
        
#         features_df = features_df.sort_values(['user_id', 'login_timestamp'])
        
        
#         features_df['time_since_last_login'] = features_df.groupby('user_id')['login_timestamp'].diff().dt.total_seconds() / 3600
#         features_df['time_since_last_login'] = features_df['time_since_last_login'].fillna(24)  # Default 24 hours
        
       
#         features_df['is_rapid_login'] = (features_df['time_since_last_login'] < 0.2).astype(int)  # <6 minutes
#         features_df['is_very_delayed_login'] = (features_df['time_since_last_login'] > 12).astype(int)  # >1 week
        
        
#         user_stats = features_df.groupby('user_id').agg({
#             'login_timestamp': ['count', 'nunique'],
#             'hour': ['mean', 'std'],
#             'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
#         }).round(2)
        
#         user_stats.columns = ['total_logins', 'unique_days', 'avg_hour', 'hour_std', 'typical_day']
#         user_stats = user_stats.reset_index()
        
#         features_df = features_df.merge(user_stats, on='user_id', how='left')
        
#         features_df['hour_deviation'] = np.abs(features_df['hour'] - features_df['avg_hour'])
#         features_df['day_deviation'] = np.abs(features_df['day_of_week'] - features_df['typical_day'])
        
#         for col in ['device_type', 'browser']:
#             if col in features_df.columns:
#                 user_device_counts = features_df.groupby('user_id')[col].nunique()
#                 features_df[f'{col}_diversity'] = features_df['user_id'].map(user_device_counts)
                
#                 user_common = features_df.groupby('user_id')[col].apply(
#                     lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
#                 )
#                 features_df[f'usual_{col}'] = features_df['user_id'].map(user_common)
#                 features_df[f'is_usual_{col}'] = (features_df[col] == features_df[f'usual_{col}']).astype(int)
        
#         return features_df
    
#     def create_balanced_labels(self, df):
#         risk_weights = {
#             'is_blacklisted': 3.0,  
#             'login_success': 2.0,  
#             'is_off_hours': 1.0,    
#             'distance_from_usual': 1.5,  
#             'is_new_location': 2.0,
#             'is_rapid_login': 1.0,  
#             'hour_deviation': 1.5,  
#             'ip_rarity': 1.0      
#         }
        
#         risk_score = np.zeros(len(df))
        
#         for feature, weight in risk_weights.items():
#             if feature in df.columns:
#                 if feature == 'login_success':
#                     risk_score += weight * (1 - df[feature].fillna(1))
#                 elif feature == 'distance_from_usual':
#                     threshold = df[feature].quantile(0.9)
#                     risk_score += weight * (df[feature] > threshold).astype(int)
#                 elif feature == 'hour_deviation':
#                     threshold = df[feature].quantile(0.8)
#                     risk_score += weight * (df[feature] > threshold).astype(int)
#                 elif feature == 'ip_rarity':
#                     threshold = df[feature].quantile(0.9)
#                     risk_score += weight * (df[feature] > threshold).astype(int)
#                 else:
#                     risk_score += weight * df[feature].fillna(0)
        
#         threshold = np.percentile(risk_score, 65)  
#         labels = (risk_score >= threshold).astype(int)
        
#         print(f"Created labels with {labels.sum()} suspicious cases ({labels.mean():.1%})")
#         return labels, risk_score
    
#     def prepare_features_enhanced(self, df, fit_encoders=True):
        
#         numeric_features = [
#             'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
#             'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
#             'distance_from_usual', 'location_consistency', 'is_new_location',
#             'latitude', 'longitude', 'ip_frequency', 'ip_rarity', 'user_ip_diversity',
#             'subnet_frequency', 'time_since_last_login', 'is_rapid_login', 'is_very_delayed_login',
#             'total_logins', 'avg_hour', 'hour_std', 'hour_deviation', 'day_deviation',
#             'device_type_diversity', 'browser_diversity', 'is_usual_device_type', 'is_usual_browser',
#             'pincode', 'is_blacklisted', 'login_success', 'is_off_hours'
#         ]
        
#         feature_columns = []
#         for feature in numeric_features:
#             if feature in df.columns:
#                 feature_columns.append(feature)
        

#         categorical_features = ['device_type', 'browser', 'ip_subnet']
#         for feature in categorical_features:
#             if feature in df.columns:
#                 if fit_encoders:
#                     self.label_encoders[feature] = LabelEncoder()
#                     df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
#                         df[feature].fillna('Unknown').astype(str)
#                     )
#                 else:
#                     if feature in self.label_encoders:
#                         df[feature] = df[feature].fillna('Unknown').astype(str)
#                         known_categories = set(self.label_encoders[feature].classes_)
#                         df[feature] = df[feature].apply(
#                             lambda x: x if x in known_categories else 'Unknown'
#                         )
#                         df[feature] = df[feature].fillna("Unknown").astype(str)
#                         known_categories = set(self.label_encoders[feature].classes_)
#                         df[feature] = df[feature].apply(lambda x: x if x in known_categories else "Unknown")
#                         df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
#                     else:
#                         df[f'{feature}_encoded'] = 0
#                 feature_columns.append(f'{feature}_encoded')

#         available_features = [f for f in feature_columns if f in df.columns]
#         X = df[available_features].fillna(0)
        
#         if fit_encoders:
#             self.feature_names = available_features
#             X_scaled = self.scaler.fit_transform(X)
#         else:
#             X_scaled = self.scaler.transform(X)
        
#         return X_scaled, available_features
    
#     def train_with_resampling(self, df, use_resampling=True):
        
#         df_features = self.enhanced_feature_engineering(df)
        
#         y, risk_scores = self.create_balanced_labels(df_features)
        
#         X, feature_names = self.prepare_features_enhanced(df_features, fit_encoders=True)
        
#         print(f"Training with {len(X)} samples, {len(feature_names)} features")
#         print(f"Class distribution: {np.bincount(y)}")
        
#         if use_resampling and y.sum() > 0:  
#             try:
#                 smote = SMOTE(random_state=42, k_neighbors=min(5, y.sum()-1))
#                 X, y = smote.fit_resample(X, y)
#                 print(f"After SMOTE: {np.bincount(y)}")
#             except:

#                 from imblearn.over_sampling import RandomOverSampler
#                 ros = RandomOverSampler(random_state=42)
#                 X, y = ros.fit_resample(X, y)
#                 print(f"After Random Oversampling: {np.bincount(y)}")
        

#         self.model.fit(X, y)
        
#         return df_features, y
    
#     def predict_enhanced(self, df):
#         df_features = self.enhanced_feature_engineering(df)
#         X, _ = self.prepare_features_enhanced(df_features, fit_encoders=False) #false
        
#         predictions = self.model.predict(X)
#         proba = self.model.predict_proba(X)
#         scores = proba[:, 1] 
        
#         return predictions, scores, df_features
    
#     def optimize_threshold(self, y_true, scores):
#         precision, recall, thresholds = precision_recall_curve(y_true, scores)
        
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#         best_idx = np.argmax(f1_scores)
#         best_threshold = thresholds[best_idx]
        
#         print(f"Optimal threshold: {best_threshold:.4f}")
#         print(f"Best F1 score: {f1_scores[best_idx]:.4f}")
        
#         return best_threshold
    
#     def cross_validate_model(self, df, cv_folds=5):
#         df_features = self.enhanced_feature_engineering(df)
#         y, _ = self.create_balanced_labels(df_features)
#         X, _ = self.prepare_features_enhanced(df_features, fit_encoders=True)
        
#         if y.sum() > 0:
#             cv_scores = cross_val_score(
#                 self.model, X, y, 
#                 cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
#                 scoring='f1'
#             )
#             print(f"Cross-validation F1 scores: {cv_scores}")
#             print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
#             return cv_scores
#         else:
#             print("No positive cases for cross-validation")
#             return None
#     def save_model(self, filepath=None):
#         if filepath is None:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filepath = f"suspicious_login_model_{timestamp}.joblib"
        
#         model_data = {
#             'model': self.model,
#             'label_encoders': self.label_encoders,
#             'scaler': self.scaler,
#             'feature_names': self.feature_names,
#             'contamination': self.contamination,
#             'method': self.method
#         }
        
#         joblib.dump(model_data, filepath)
#         print(f"Model saved to: {filepath}")
#         return filepath
    
#     @classmethod
#     def load_model(cls, filepath):
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Model file not found: {filepath}")
        
#         model_data = joblib.load(filepath)
        

#         detector = cls(contamination=model_data.get('contamination', 0.1))
        

#         detector.model = model_data['model']
#         detector.label_encoders = model_data['label_encoders']
#         detector.scaler = model_data['scaler']
#         detector.feature_names = model_data['feature_names']
#         detector.method = model_data.get('method', 'logistic_regression')
        
#         print(f"Model loaded from: {filepath}")
#         return detector

# def run_improved_pipeline(csv_file_path, use_resampling=True):
   
#     detector = ImprovedSuspiciousLoginDetector(contamination=0.15)
    
   
#     try:
#         df = pd.read_csv(csv_file_path)
#         print(f"Loaded {len(df)} records")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None, None
    

#     df = df.dropna(subset=['user_id'])
#     if 'login_timestamp' in df.columns:
#         df['login_timestamp'] = pd.to_datetime(df['login_timestamp'])
    

#     train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
#     print(f"Training: {len(train_df)}, Testing: {len(test_df)}")

#     train_features, train_labels = detector.train_with_resampling(train_df, use_resampling)
    
#     detector.cross_validate_model(train_df)
    
#     predictions, scores, test_features = detector.predict_enhanced(test_df)
    
#     test_labels, _ = detector.create_balanced_labels(test_features)
    
#     optimal_threshold = detector.optimize_threshold(test_labels, scores)
#     joblib.dump(optimal_threshold, "threshold.joblib")
#     print(f"Optimal threshold saved to threshold.joblib")

#     predictions = (scores >= optimal_threshold).astype(int)
    
#     print("="*50)
#     print("RESULTS")
#     print("="*50)
#     print(confusion_matrix(test_labels, predictions))
#     print(classification_report(test_labels, predictions, target_names=['Normal', 'Suspicious']))
    
#     if len(np.unique(test_labels)) > 1:
#         auc_score = roc_auc_score(test_labels, scores)
#         print(f"\nAUC-ROC Score: {auc_score:.4f}")
    
#     return detector, test_features


# def save_and_load_example():
#     detector, results = run_improved_pipeline("login_data_with_geo.csv")
    
#     model_path = detector.save_model("my_login_detector.joblib")
    
#     loaded_detector = ImprovedSuspiciousLoginDetector.load_model("my_login_detector.joblib")
    
   
# if __name__ == "__main__":
#     csv_file_path = "login_data_with_geo.csv"
    
#     print(f"\n{'='*20} TESTING LOGISTIC_REGRESSION {'='*20}")
#     detector, results = run_improved_pipeline(csv_file_path)
    
#     if detector:
#         model_path = detector.save_model()
#         print(f"Model training completed and saved!")
        
        
#         print("\nTesting model loading...")
#         loaded_detector = ImprovedSuspiciousLoginDetector.load_model(model_path)
#         print("Model loaded successfully!")
    
#     print("\n" + "="*60)



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
