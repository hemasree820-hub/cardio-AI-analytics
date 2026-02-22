# 🏥 Cardiovascular Risk Analytics Suite
### *Bridging Clinical Expertise and Explainable AI*

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo
**[View the Live App Here](https://hemasree820-hub-cardio-ai-analytics-app.streamlit.app/)**

## 📖 Project Overview
As a **Perfusionist**, I noticed a gap in how clinical teams interact with predictive data. Many AI models are "black boxes," making it hard for clinicians to trust them in high-stakes environments.

This application is a **Clinical Decision Support Tool** designed to predict patient ischemic risk and postoperative complications. It doesn't just provide a score; it uses **SHAP (Explainable AI)** to show *exactly* which clinical parameters (MAP, Hemoglobin, Lactate) are driving the risk.

## ✨ Key Features
* **Predictive Modeling:** Forecasts cardiovascular risk scores based on intraoperative data.
* **Explainable AI (XAI):** Integrated SHAP visualizations to provide transparency for every prediction.
* **Clinical Correlation Heatmaps:** Interactive analysis of the relationship between vital signs and patient outcomes.
* **Dynamic UI:** Built with Streamlit for real-time data exploration.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-Learn (Random Forest/XGBoost)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Data Visualization:** Plotly & Seaborn

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application
├── model.py            # Machine learning logic and training
├── requirements.txt    # List of dependencies
├── processed_data.csv  # Cleaned dataset (Synthetic/Anonymized)
└── README.md           # Project documentation
