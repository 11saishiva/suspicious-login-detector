from logistic_regression import run_improved_pipeline
import joblib

detector, threshold = run_improved_pipeline("login_data_with_geo.csv")

joblib.dump(threshold, "threshold.joblib")
print(f"Saved threshold:\n {threshold}")
