# 🌿 AyushSetu – Modernizing Panchakarma Progress Tracking

**AyushSetu** is an AI-powered system designed to **modernize the Ayurvedic Panchakarma consultation and progress tracking process**.
It combines **traditional Ayurvedic insights** with **machine learning models** to help both doctors and patients save time, track improvements, and visualize Vikriti (dosha imbalance) levels.

---

## ✨ Features

* 🩺 **Therapy-specific Models** – 5 Panchakarma therapies, each with a dedicated ML model.
* 🌱 **General Health Model** – A unified model to predict overall wellness improvement across all therapies.
* 📊 **MultiOutput Regression** – Predicts:

  * Vikriti (dosha imbalance) levels in **percentage**
  * Overall therapy improvement
  * General improvement
* ⚡ **XGBoost (XGBRegressor)** with `MultiOutputRegressor` for accurate predictions.
* 🌐 **Flask API Integration** – Interactive consultation forms connected to ML models.
* 🖥 **User-friendly HTML Forms** – Separate forms for each therapy, divided into **Mental** and **Physical health** sections for clarity.
* 🔄 **Progress Tracking** – Patients and doctors can track therapy impact over multiple sessions.

---

## 📚 Panchakarma Therapies Covered

The system currently supports:

1. **Vamana** – Detoxification through emesis
2. **Virechana** – Detoxification through purgation
3. **Basti** – Enema therapy
4. **Nasya** – Nasal detox
5. **Raktamokshana** – Bloodletting therapy

Each therapy has its own tailored consultation form + prediction model.

---

## 🛠 Tech Stack

* **Frontend**: HTML5, CSS3 (beautiful Ayurveda-inspired UI)
* **Backend**: Flask (Python API)
* **ML Models**: XGBRegressor + MultiOutputRegressor
* **Data**: Synthetic data generated based on Ayurvedic patterns & general trends
* **Deployment Ready**: Can be extended to cloud hosting

---

## 🚀 How It Works

1. **Doctor/Patient opens therapy form**

   * General health questions (mood, stress, digestion, sleep, etc.)
   * Therapy-specific questions (unique to Panchakarma method)
2. **Flask API processes input**

   * Passes general inputs → **General Model**
   * Passes all inputs → **Therapy-Specific Model**
3. **Predictions Returned**

   * Vikriti (Dosha Imbalance %)
   * Overall Therapy Improvement
   * General Well-being Improvement
4. **Results Displayed** in a clean, Ayurveda-inspired dashboard.

---



## 🧪 Example Output

```json
{
  "therapy": "Raktamokshana",
  "vikriti_levels": {
    "Pitta": "67%",
    "Vata": "21%",
    "Kapha": "12%"
  },
  "overall_therapy_improvement": "Moderate Improvement",
  "general_improvement": "High"
}
```

---

## 🌍 Vision

**AyushSetu** bridges the gap between **ancient wisdom** and **modern technology**.
It empowers Ayurvedic practitioners with **data-driven insights** while giving patients a **faster, personalized, and trackable** wellness journey.

---

🔮 Future Enhancements
📈 Historical tracking with graphs (progress over time)
☁️ Cloud deployment for multi-clinic use
📱 Mobile-first responsive UI
🤝 Integration with EHR/Patient apps
🧘 Dosha-specific lifestyle recommendations
🧠 GRU-based Neural Network – Predict next 2 days of therapy improvement trends
---

💚 *“Balancing ancient Ayurveda with modern AI – one therapy at a time.”*
