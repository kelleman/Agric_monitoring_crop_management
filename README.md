# 🌾 Smart Agriculture Monitoring and Crop Management System

An **autonomous framework** for agricultural monitoring and crop recommendation using **machine learning**. This system intelligently suggests the best crop to cultivate based on soil nutrients and weather conditions, while also offering **fertilizer suggestions**, **pest alerts**, and **irrigation advice** — all through an intuitive **web interface powered by Streamlit**.

## 💡 Features

- ✅ Crop recommendation based on soil and climate conditions.
- 🧪 Fertilizer suggestions based on NPK levels.
- 🛡️ Pest risk alerts based on weather data.
- 🚿 Irrigation guidance based on rainfall patterns.
- 📊 Visual insight into model accuracy and predictions.

## 🧰 Tech Stack

- **Python**
- **Pandas** for data handling
- **Scikit-learn** for training the machine learning model
- **RandomForestClassifier** as the core ML algorithm
- **Joblib** for model serialization
- **Streamlit** for the web app interface
- **Matplotlib / Seaborn** for data visualization

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/kelleman/Agric_monitoring_crop_management.git
```
### 2. Prepare the Data set
		Place your dataset at:
		data/crop_data.csv
		Ensure it contains the columns: N, P, K, temperature, humidity, ph, rainfall, and Crop.

### 3. Install dependencies
		run 'pip -r install requirements.txt'

### 4. Train model
		run  'python train_model.py'

### 5. Run the web App
		run 'streamlit run app.py'

