from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models
base_dir = Path(__file__).resolve().parent
with open(base_dir / "Basti_model.pkl", "rb") as f:
    basti_model = pickle.load(f)

with open(base_dir / "General_model.pkl", "rb") as f:
    general_model = pickle.load(f)

# Feature lists
basti_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings", "Mood",
    "Bowel_Dryness", "Gas_Formation", "Lower_Back_Pain",
    "Urinary_Frequency", "Constipation_Level"
]

general_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings","Mood"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("Basti_form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form values safely
        input_data = {f: [request.form.get(f, "Average")] for f in basti_features}

        df_basti = pd.DataFrame(input_data)
        df_general = pd.DataFrame({f: input_data[f] for f in general_features})

        # Predictions
        basti_pred = basti_model.predict(df_basti)
        general_pred = general_model.predict(df_general)

        vata_level = basti_pred[0] if basti_pred.ndim == 1 else basti_pred[0][0]
        overall_improvement = basti_pred[0][1] if (basti_pred.ndim == 2 and basti_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]

        return render_template(
            "result_basti.html",
            vata_level=vata_level,
            overall_improvement=overall_improvement,
            general_improvement=general_improvement
        )
    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
