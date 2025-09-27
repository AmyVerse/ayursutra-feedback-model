from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models
base_dir = Path(__file__).resolve().parent
with open(base_dir / "Vamana_model.pkl", "rb") as f:
    vamana_model = pickle.load(f)

with open(base_dir / "General_model.pkl", "rb") as f:
    general_model = pickle.load(f)

# Feature lists for each model
vamana_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood", "Thirst_Level",
    "Mood_Swings", "Body_Temperature", "Metabolism", "Immunity"
]

general_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings", "Mood"
]


@app.route("/", methods=["GET"])
def index():
    return render_template("Vamana_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form values as strings (no conversion to float)
        input_data = {feature: [request.form[feature]] for feature in vamana_features}

        # Create DataFrames for each model
        df_vamana = pd.DataFrame({feature: input_data[feature] for feature in vamana_features})
        df_general = pd.DataFrame({feature: input_data[feature] for feature in general_features})

        # Predictions
        vamana_pred = vamana_model.predict(df_vamana)
        general_pred = general_model.predict(df_general)

        # Handle vamana prediction (2D vs 1D output)
        if hasattr(vamana_pred, "ndim") and vamana_pred.ndim == 2 and vamana_pred.shape[1] >= 2:
            kapha_level = vamana_pred[0][0]
            overall_improvement = vamana_pred[0][1]
        else:
            kapha_level = vamana_pred[0]
            overall_improvement = None

        general_improvement = general_pred[0]

        return render_template(
            "result_vamana.html",
            kapha_level=kapha_level,
            overall_improvement=overall_improvement,
            general_improvement=general_improvement
        )

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"


if __name__ == "__main__":
    app.run(debug=True)
