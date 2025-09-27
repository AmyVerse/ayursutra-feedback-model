from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models
base_dir = Path(__file__).resolve().parent
with open(base_dir / "Raktamokshana_model.pkl", "rb") as f:
    raktamokshana_model = pickle.load(f)

with open(base_dir / "General_model.pkl", "rb") as f:
    general_model = pickle.load(f)

# Features from the Raktamokshana.html form
general_features = [
    "Mood", "Stress_Level", "Concentration", "Mood_Swings",
    "Energy_Level", "Appetite", "Sleep_Quality", "Digestion",
    "Physical_Activity", "Hydration"
]

raktamokshana_features = [
    "Mood", "Stress_Level", "Concentration", "Mood_Swings",
    "Energy_Level", "Appetite", "Sleep_Quality", "Digestion",
    "Physical_Activity", "Hydration",
    "Skin_Redness", "Body_Heat", "Acidity", "Bleeding_Tendency", "Inflammation"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("Raktamokshana.html")


@app.route("/predict_raktamokshana", methods=["POST"])
def predict_raktamokshana():
    try:
        # Collect all form values
        input_data = {feature: [request.form[feature]] for feature in general_features + raktamokshana_features}

        # Create DataFrames
        df_general = pd.DataFrame({feature: input_data[feature] for feature in general_features})
        df_raktamokshana = pd.DataFrame({feature: input_data[feature] for feature in raktamokshana_features})

        # Predictions
        general_pred = general_model.predict(df_general)
        raktamokshana_pred = raktamokshana_model.predict(df_raktamokshana)

        # Handle Raktamokshana model output (2D vs 1D)
        if hasattr(raktamokshana_pred, "ndim") and raktamokshana_pred.ndim == 2 and raktamokshana_pred.shape[1] >= 2:
            pitta_level = raktamokshana_pred[0][0]
            overall_improvement = raktamokshana_pred[0][1]
        else:
            pitta_level = raktamokshana_pred[0]
            overall_improvement = None

        general_improvement = general_pred[0]

        return render_template(
            "result_raktamokshana.html",
            pitta_level=pitta_level,
            overall_improvement=overall_improvement,
            general_improvement=general_improvement
        )

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"


if __name__ == "__main__":
    app.run(debug=True)
