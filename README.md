# ☢️ War Outcome Awareness - India Pakistan Conflict

A Machine Learning-powered Flask web app that simulates the **impact of nuclear conflict** between India and Pakistan. The project models **nuclear winter scenarios**, predicts potential consequences using trained ML models, and provides **data visualizations** and an embedded chatbot interface.

---

## 🌐 Live Preview

🚀 _Coming soon (hosted on Vercel, Railway, or local server)_

---

## 📌 Features

- 🔮 **Predictive Simulation**: Forecasts outcomes of nuclear winter using pre-trained models
- 🧠 **ML-Backed Insights**: Trained using Random Forest & XGBoost on synthetic nuclear data
- 📊 **Data Visualizations**: Includes distribution plots, pairplots, and temperature projections
- 💬 **Integrated Chatbot**: Gemini-based chatbot for geopolitical queries
- 📁 **Git LFS-Optimized**: Stores large models and images via Git LFS

---

## 🛠️ Tech Stack

| Layer         | Technologies                     |
|---------------|----------------------------------|
| Frontend      | HTML5, CSS, Jinja2 (Flask)       |
| Backend       | Python, Flask                    |
| ML Models     | Scikit-learn, Joblib             |
| Visuals       | Matplotlib, Seaborn              |
| Deployment    | GitHub + Git LFS                 |

---

## 📂 Project Structure

```plaintext
WarOutcomeAwarenessV2/
│
├── static/                        # Static assets (plots/images)
│   ├── boxplot.png
│   ├── distribution.png
│   ├── pairplot.png
│   └── soot_temp.png
│
├── templates/                    # HTML Templates (Jinja2)
│   ├── index.html
│   ├── predict.html
│   ├── chatbot.html
│   ├── error.html
│
├── best_model_name.joblib       # Trained ML model (LFS)
├── feature_scaler.joblib        # Preprocessing scaler (LFS)
├── nuclear_winter_simulations.csv  # Dataset (LFS)
├── app.py                       # Main Flask application
├── requirements.txt             # Python dependencies
└── .gitattributes               # Git LFS tracking config
