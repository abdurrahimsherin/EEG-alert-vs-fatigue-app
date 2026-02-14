# fatigue_predict.py
import numpy as np
import pandas as pd
import joblib

# Paths to saved model/scaler/encoder
MODEL_PATH = r"E:\My_project\models\svm_models\fold_1\svm_model.pkl"
SCALER_PATH = r"E:\My_project\models\svm_models\fold_1\scaler.pkl"
ENCODER_PATH = r"E:\My_project\models\svm_models\label_encoder.pkl"

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)


def predict_single(features_str: str):
    """Predict from comma-separated feature string"""
    try:
        features_list = [float(x.strip()) for x in features_str.split(',')]
        features_array = np.array(features_list).reshape(1, -1)

        features_scaled = scaler.transform(features_array)
        pred_class = model.predict(features_scaled)[0]
        pred_prob = model.predict_proba(features_scaled)[0][1]

        pred_label = le.inverse_transform([pred_class])[0]
        return pred_label, float(pred_prob)

    except Exception as e:
        return f"Error: {str(e)}", None


def predict_file(file_path):
    """Predict fatigue for each row in uploaded CSV"""
    try:
        df = pd.read_csv(file_path)

        # Ensure only numeric features are used
        X = df.select_dtypes(include=[np.number])

        if X.empty:
            return "Error: No numeric columns found in file.", None

        # Scale and predict
        X_scaled = scaler.transform(X.values)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        labels = le.inverse_transform(preds)

        # Add results to dataframe
        df["Predicted_Label"] = labels
        df["Fatigue_Probability"] = probs

        return df, None

    except Exception as e:
        return f"Error: {str(e)}", None
