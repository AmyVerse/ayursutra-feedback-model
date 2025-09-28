from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import pickle
import pandas as pd
import os
from pathlib import Path
from typing import Optional

app = FastAPI(title="AyurSutra Feedback Model", description="AI-powered Panchakarma therapy feedback system")

# Load all models
base_dir = Path(__file__).resolve().parent
models = {}
try:
    with open(base_dir / "Basti_model.pkl", "rb") as f:
        models["basti"] = pickle.load(f)
    with open(base_dir / "Nasya_model.pkl", "rb") as f:
        models["nasya"] = pickle.load(f)
    with open(base_dir / "Vamana_model.pkl", "rb") as f:
        models["vamana"] = pickle.load(f)
    with open(base_dir / "Virechana_model.pkl", "rb") as f:
        models["virechana"] = pickle.load(f)
    with open(base_dir / "Raktamokshana_model.pkl", "rb") as f:
        models["raktamokshana"] = pickle.load(f)
    with open(base_dir / "General_model.pkl", "rb") as f:
        models["general"] = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

templates = Jinja2Templates(directory="templates")

# Feature lists for each therapy
therapy_features = {
    "basti": [
        "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
        "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
        "Hydration", "Mood_Swings", "Mood",
        "Bowel_Dryness", "Gas_Formation", "Lower_Back_Pain",
        "Urinary_Frequency", "Constipation_Level"
    ],
    "nasya": [
        "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
        "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
        "Hydration", "Mood_Swings", "Mood",
        "Nasal_Dryness", "Headache", "Dizziness", "Sinus_Congestion", "Throat_Dryness"
    ],
    "vamana": [
        "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
        "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
        "Hydration", "Mood_Swings", "Mood",
        "Nausea", "Acidity", "Weight_Gain", "Food_Intolerance", "Bloating"
    ],
    "virechana": [
        "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
        "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
        "Hydration", "Mood_Swings", "Mood",
        "Acidity", "Heartburn", "Skin_Issues", "Body_Heat", "Irritability"
    ],
    "raktamokshana": [
        "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
        "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
        "Hydration", "Mood_Swings", "Mood",
        "Skin_Redness", "Joint_Pain", "Inflammation", "Blood_Pressure", "Circulation"
    ]
}

general_features = [
    "Concentration", "Sleep_Quality", "Digestion", "Flexibility",
    "Energy_Level", "Appetite", "Stress_Level", "Physical_Activity",
    "Hydration", "Mood_Swings", "Mood"
]

# Global variable to store last results (in production, use sessions or database)
last_results = {}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page with therapy selection"""
    return templates.TemplateResponse("index.html", {"request": request})

# Basti therapy routes
@app.get("/basti", response_class=HTMLResponse)
def basti_form(request: Request):
    return templates.TemplateResponse("Basti_form.html", {"request": request})

@app.post("/basti/predict", response_class=HTMLResponse)
def predict_basti(
    request: Request,
    Concentration: str = Form(...),
    Sleep_Quality: str = Form(...),
    Digestion: str = Form(...),
    Flexibility: str = Form("Average"),
    Energy_Level: str = Form(...),
    Appetite: str = Form(...),
    Stress_Level: str = Form(...),
    Physical_Activity: str = Form(...),
    Hydration: str = Form(...),
    Mood_Swings: str = Form(...),
    Mood: str = Form(...),
    Bowel_Dryness: str = Form(...),
    Gas_Formation: str = Form(...),
    Lower_Back_Pain: str = Form(...),
    Urinary_Frequency: str = Form(...),
    Constipation_Level: str = Form(...)
):
    try:
        form_data = {
            "Concentration": [Concentration],
            "Sleep_Quality": [Sleep_Quality],
            "Digestion": [Digestion],
            "Flexibility": [Flexibility],
            "Energy_Level": [Energy_Level],
            "Appetite": [Appetite],
            "Stress_Level": [Stress_Level],
            "Physical_Activity": [Physical_Activity],
            "Hydration": [Hydration],
            "Mood_Swings": [Mood_Swings],
            "Mood": [Mood],
            "Bowel_Dryness": [Bowel_Dryness],
            "Gas_Formation": [Gas_Formation],
            "Lower_Back_Pain": [Lower_Back_Pain],
            "Urinary_Frequency": [Urinary_Frequency],
            "Constipation_Level": [Constipation_Level]
        }
        
        df_basti = pd.DataFrame(form_data)
        df_general = pd.DataFrame({f: form_data[f] for f in general_features})
        
        basti_pred = models["basti"].predict(df_basti)
        general_pred = models["general"].predict(df_general)
        
        vata_level = basti_pred[0] if basti_pred.ndim == 1 else basti_pred[0][0]
        overall_improvement = basti_pred[0][1] if (basti_pred.ndim == 2 and basti_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]
        
        global last_results
        last_results = {
            "therapy": "Basti",
            "vata_level": vata_level,
            "overall_improvement": overall_improvement,
            "general_improvement": general_improvement
        }
        
        return RedirectResponse(url="/results/basti", status_code=303)
        
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>❌ Error: {e}</h3>")

# Nasya therapy routes
@app.get("/nasya", response_class=HTMLResponse)
def nasya_form(request: Request):
    return templates.TemplateResponse("Nasya_form.html", {"request": request})

@app.post("/nasya/predict", response_class=HTMLResponse)
def predict_nasya(
    request: Request,
    Concentration: str = Form(...),
    Sleep_Quality: str = Form(...),
    Digestion: str = Form(...),
    Flexibility: str = Form("Average"),
    Energy_Level: str = Form(...),
    Appetite: str = Form(...),
    Stress_Level: str = Form(...),
    Physical_Activity: str = Form(...),
    Hydration: str = Form(...),
    Mood_Swings: str = Form(...),
    Mood: str = Form(...),
    Nasal_Dryness: str = Form(...),
    Headache: str = Form(...),
    Dizziness: str = Form(...),
    Sinus_Congestion: str = Form(...),
    Throat_Dryness: str = Form(...)
):
    try:
        form_data = {
            "Concentration": [Concentration],
            "Sleep_Quality": [Sleep_Quality],
            "Digestion": [Digestion],
            "Flexibility": [Flexibility],
            "Energy_Level": [Energy_Level],
            "Appetite": [Appetite],
            "Stress_Level": [Stress_Level],
            "Physical_Activity": [Physical_Activity],
            "Hydration": [Hydration],
            "Mood_Swings": [Mood_Swings],
            "Mood": [Mood],
            "Nasal_Dryness": [Nasal_Dryness],
            "Headache": [Headache],
            "Dizziness": [Dizziness],
            "Sinus_Congestion": [Sinus_Congestion],
            "Throat_Dryness": [Throat_Dryness]
        }
        
        df_nasya = pd.DataFrame(form_data)
        df_general = pd.DataFrame({f: form_data[f] for f in general_features})
        
        nasya_pred = models["nasya"].predict(df_nasya)
        general_pred = models["general"].predict(df_general)
        
        vata_level = nasya_pred[0] if nasya_pred.ndim == 1 else nasya_pred[0][0]
        overall_improvement = nasya_pred[0][1] if (nasya_pred.ndim == 2 and nasya_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]
        
        global last_results
        last_results = {
            "therapy": "Nasya",
            "vata_level": vata_level,
            "overall_improvement": overall_improvement,
            "general_improvement": general_improvement
        }
        
        return RedirectResponse(url="/results/nasya", status_code=303)
        
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>❌ Error: {e}</h3>")

# Vamana therapy routes
@app.get("/vamana", response_class=HTMLResponse)
def vamana_form(request: Request):
    return templates.TemplateResponse("Vamana_form.html", {"request": request})

@app.post("/vamana/predict", response_class=HTMLResponse)
def predict_vamana(
    request: Request,
    Concentration: str = Form(...),
    Sleep_Quality: str = Form(...),
    Digestion: str = Form(...),
    Flexibility: str = Form("Average"),
    Energy_Level: str = Form(...),
    Appetite: str = Form(...),
    Stress_Level: str = Form(...),
    Physical_Activity: str = Form(...),
    Hydration: str = Form(...),
    Mood_Swings: str = Form(...),
    Mood: str = Form(...),
    Nausea: str = Form(...),
    Acidity: str = Form(...),
    Weight_Gain: str = Form(...),
    Food_Intolerance: str = Form(...),
    Bloating: str = Form(...)
):
    try:
        form_data = {
            "Concentration": [Concentration],
            "Sleep_Quality": [Sleep_Quality],
            "Digestion": [Digestion],
            "Flexibility": [Flexibility],
            "Energy_Level": [Energy_Level],
            "Appetite": [Appetite],
            "Stress_Level": [Stress_Level],
            "Physical_Activity": [Physical_Activity],
            "Hydration": [Hydration],
            "Mood_Swings": [Mood_Swings],
            "Mood": [Mood],
            "Nausea": [Nausea],
            "Acidity": [Acidity],
            "Weight_Gain": [Weight_Gain],
            "Food_Intolerance": [Food_Intolerance],
            "Bloating": [Bloating]
        }
        
        df_vamana = pd.DataFrame(form_data)
        df_general = pd.DataFrame({f: form_data[f] for f in general_features})
        
        vamana_pred = models["vamana"].predict(df_vamana)
        general_pred = models["general"].predict(df_general)
        
        kapha_level = vamana_pred[0] if vamana_pred.ndim == 1 else vamana_pred[0][0]
        overall_improvement = vamana_pred[0][1] if (vamana_pred.ndim == 2 and vamana_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]
        
        global last_results
        last_results = {
            "therapy": "Vamana",
            "kapha_level": kapha_level,
            "overall_improvement": overall_improvement,
            "general_improvement": general_improvement
        }
        
        return RedirectResponse(url="/results/vamana", status_code=303)
        
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>❌ Error: {e}</h3>")

# Virechana therapy routes
@app.get("/virechana", response_class=HTMLResponse)
def virechana_form(request: Request):
    return templates.TemplateResponse("Virechana_form.html", {"request": request})

@app.post("/virechana/predict", response_class=HTMLResponse)
def predict_virechana(
    request: Request,
    Concentration: str = Form(...),
    Sleep_Quality: str = Form(...),
    Digestion: str = Form(...),
    Flexibility: str = Form("Average"),
    Energy_Level: str = Form(...),
    Appetite: str = Form(...),
    Stress_Level: str = Form(...),
    Physical_Activity: str = Form(...),
    Hydration: str = Form(...),
    Mood_Swings: str = Form(...),
    Mood: str = Form(...),
    Acidity: str = Form(...),
    Heartburn: str = Form(...),
    Skin_Issues: str = Form(...),
    Body_Heat: str = Form(...),
    Irritability: str = Form(...)
):
    try:
        form_data = {
            "Concentration": [Concentration],
            "Sleep_Quality": [Sleep_Quality],
            "Digestion": [Digestion],
            "Flexibility": [Flexibility],
            "Energy_Level": [Energy_Level],
            "Appetite": [Appetite],
            "Stress_Level": [Stress_Level],
            "Physical_Activity": [Physical_Activity],
            "Hydration": [Hydration],
            "Mood_Swings": [Mood_Swings],
            "Mood": [Mood],
            "Acidity": [Acidity],
            "Heartburn": [Heartburn],
            "Skin_Issues": [Skin_Issues],
            "Body_Heat": [Body_Heat],
            "Irritability": [Irritability]
        }
        
        df_virechana = pd.DataFrame(form_data)
        df_general = pd.DataFrame({f: form_data[f] for f in general_features})
        
        virechana_pred = models["virechana"].predict(df_virechana)
        general_pred = models["general"].predict(df_general)
        
        pitta_level = virechana_pred[0] if virechana_pred.ndim == 1 else virechana_pred[0][0]
        overall_improvement = virechana_pred[0][1] if (virechana_pred.ndim == 2 and virechana_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]
        
        global last_results
        last_results = {
            "therapy": "Virechana",
            "pitta_level": pitta_level,
            "overall_improvement": overall_improvement,
            "general_improvement": general_improvement
        }
        
        return RedirectResponse(url="/results/virechana", status_code=303)
        
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>❌ Error: {e}</h3>")

# Raktamokshana therapy routes
@app.get("/raktamokshana", response_class=HTMLResponse)
def raktamokshana_form(request: Request):
    return templates.TemplateResponse("Raktamokshana.html", {"request": request})

@app.post("/raktamokshana/predict", response_class=HTMLResponse)
def predict_raktamokshana(
    request: Request,
    Concentration: str = Form(...),
    Sleep_Quality: str = Form(...),
    Digestion: str = Form(...),
    Flexibility: str = Form("Average"),
    Energy_Level: str = Form(...),
    Appetite: str = Form(...),
    Stress_Level: str = Form(...),
    Physical_Activity: str = Form(...),
    Hydration: str = Form(...),
    Mood_Swings: str = Form(...),
    Mood: str = Form(...),
    Skin_Redness: str = Form(...),
    Joint_Pain: str = Form(...),
    Inflammation: str = Form(...),
    Blood_Pressure: str = Form(...),
    Circulation: str = Form(...)
):
    try:
        form_data = {
            "Concentration": [Concentration],
            "Sleep_Quality": [Sleep_Quality],
            "Digestion": [Digestion],
            "Flexibility": [Flexibility],
            "Energy_Level": [Energy_Level],
            "Appetite": [Appetite],
            "Stress_Level": [Stress_Level],
            "Physical_Activity": [Physical_Activity],
            "Hydration": [Hydration],
            "Mood_Swings": [Mood_Swings],
            "Mood": [Mood],
            "Skin_Redness": [Skin_Redness],
            "Joint_Pain": [Joint_Pain],
            "Inflammation": [Inflammation],
            "Blood_Pressure": [Blood_Pressure],
            "Circulation": [Circulation]
        }
        
        df_raktamokshana = pd.DataFrame(form_data)
        df_general = pd.DataFrame({f: form_data[f] for f in general_features})
        
        raktamokshana_pred = models["raktamokshana"].predict(df_raktamokshana)
        general_pred = models["general"].predict(df_general)
        
        pitta_level = raktamokshana_pred[0] if raktamokshana_pred.ndim == 1 else raktamokshana_pred[0][0]
        overall_improvement = raktamokshana_pred[0][1] if (raktamokshana_pred.ndim == 2 and raktamokshana_pred.shape[1] > 1) else None
        general_improvement = general_pred[0]
        
        global last_results
        last_results = {
            "therapy": "Raktamokshana",
            "pitta_level": pitta_level,
            "overall_improvement": overall_improvement,
            "general_improvement": general_improvement
        }
        
        return RedirectResponse(url="/results/raktamokshana", status_code=303)
        
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>❌ Error: {e}</h3>")

# Results routes
@app.get("/results/{therapy}", response_class=HTMLResponse)
def show_results(request: Request, therapy: str):
    global last_results
    if 'last_results' not in globals() or not last_results:
        return RedirectResponse(url="/")
    
    template_map = {
        "basti": "result_basti.html",
        "nasya": "result_nasya.html",
        "vamana": "result_vamana.html",
        "virechana": "result_virechana.html",
        "raktamokshana": "result_raktamokshana.html"
    }
    
    template_name = template_map.get(therapy.lower())
    if not template_name:
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            **last_results
        }
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use 0.0.0.0 for production (when PORT is set by Railway), 127.0.0.1 for local development
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)