from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models
base_dir = Path(__file__).resolve().parent
with open(base_dir / "Nasya_model.pkl", "rb") as f:
    nasya_model = pickle.load(f)

with open(base_dir / "General_model.pkl", "rb") as f:
    general_model = pickle.load(f)

nasya_features = [
    "Concentration","Sleep_Quality","Digestion","Flexibility",
    "Energy_Level","Appetite","Stress_Level","Physical_Activity",
    "Hydration","Mood_Swings","Mood",
    "Nasal_Dryness","Headache","Dizziness","Sinus_Congestion","Throat_Dryness"
]

general_features = [
    "Concentration","Sleep_Quality","Digestion","Flexibility",
    "Energy_Level","Appetite","Stress_Level","Physical_Activity",
    "Hydration","Mood_Swings","Mood"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("Nasya_form.html")

@app.route("/predict_nasya", methods=["POST"])
def predict_nasya():
    try:
        input_data = {f:[request.form.get(f,"Average")] for f in nasya_features}
        df_nasya = pd.DataFrame(input_data)
        df_general = pd.DataFrame({f:input_data[f] for f in general_features})

        nasya_pred = nasya_model.predict(df_nasya)
        general_pred = general_model.predict(df_general)

        vata_level = nasya_pred[0] if nasya_pred.ndim==1 else nasya_pred[0][0]
        overall_improvement = nasya_pred[0][1] if (nasya_pred.ndim==2 and nasya_pred.shape[1]>1) else None
        general_improvement = general_pred[0]

        return render_template("result_nasya.html",
                               vata_level=vata_level,
                               overall_improvement=overall_improvement,
                               general_improvement=general_improvement)

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
