from logistic_regression import run_improved_pipeline
import joblib

detector, threshold = run_improved_pipeline("login_data_with_geo.csv")

detector.save_model("suspicious_login_model_20250610_114330.joblib")

joblib.dump(threshold, "threshold.joblib")

print("Model and threshold saved successfully.")
