from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models
base_dir = Path(__file__).resolve().parent
with open(base_dir / "Virechana_model.pkl", "rb") as f:
    virechana_model = pickle.load(f)

with open(base_dir / "General_model.pkl", "rb") as f:
    general_model = pickle.load(f)

# Feature lists
virechana_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings", "Mood",
    "Body_Heat", "Acidity", "Bowel_Movement", "Thirst_Level", "Skin_Inflammation"
]

general_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings","Mood"
]

# Default value map (in case a field is missing)
default_values = {
    "Very Low": "Very Low",
    "Low": "Low",
    "Moderate": "Moderate",
    "High": "High",
    "Very High": "Very High",
    "Very Poor": "Very Poor",
    "Poor": "Poor",
    "Average": "Average",
    "Good": "Good",
    "Excellent": "Excellent",
    "None": "None",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Very Severe": "Very Severe",
    "Very Rare": "Very Rare",
    "Rare": "Rare",
    "Normal": "Normal",
    "Frequent": "Frequent",
    "Very Frequent": "Very Frequent"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("Virechana_form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form values safely
        input_data = {}
        for feature in virechana_features:
            val = request.form.get(feature)
            if val is None or val == "":
                val = default_values.get("Moderate")  # safe default
            input_data[feature] = [val]

        # DataFrames
        df_virechana = pd.DataFrame(input_data)
        df_general = pd.DataFrame({f: input_data[f] for f in general_features})

        # Predictions
        virechana_pred = virechana_model.predict(df_virechana)
        general_pred = general_model.predict(df_general)

        # Handle 2D predictions
        if hasattr(virechana_pred, "ndim") and virechana_pred.ndim == 2 and virechana_pred.shape[1] >= 2:
            dosha_level = virechana_pred[0][0]
            overall_improvement = virechana_pred[0][1]
        else:
            dosha_level = virechana_pred[0]
            overall_improvement = None

        general_improvement = general_pred[0]

        return render_template(
            "result_virechana.html",
            dosha_level=dosha_level,
            overall_improvement=overall_improvement,
            general_improvement=general_improvement
        )

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)

