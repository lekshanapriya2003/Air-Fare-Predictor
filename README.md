# Air Fare Predictor

Air Fare Predictor is a production-ready machine learning API that predicts airline ticket prices using multiple regression models. The system includes structured preprocessing, feature engineering, model comparison, and real-time inference via FastAPI.

Live Demo:
[https://huggingface.co/spaces/lekshanapriya/Air_Fare_Predictor](https://huggingface.co/spaces/lekshanapriya/Air_Fare_Predictor)

---

## Model Performance

| Model             | R² Score |
| ----------------- | -------- |
| Random Forest     | 0.9756   |
| Decision Tree     | 0.9754   |
| XGBoost           | 0.9675   |
| CatBoost          | 0.9660   |
| LightGBM          | 0.9606   |
| Linear Regression | 0.9002   |

Best model: Random Forest (R² = 0.9756)

---

## Features

* Comparative training of six regression algorithms
* Automated preprocessing and categorical encoding
* Distance-based feature engineering using geospatial coordinates
* Smart airline fallback logic for incomplete inputs
* FastAPI-based REST API with interactive documentation
* Docker-ready deployment
* Hugging Face Space deployment

---

## Tech Stack

* Python 3.11
* FastAPI, Uvicorn
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Pandas, NumPy
* Geopy
* Docker

---

## Project Structure

```
air_fare_predictor/
│
├── app.py
├── Dockerfile
├── requirements.txt
│
├── src/
│   ├── predict.py
│   ├── model.py
│   ├── data.py
│   └── utils.py
│
└── models/
    ├── random_forest.pkl
    ├── decision_tree.pkl
    ├── xgboost.pkl
    ├── lightgbm.pkl
    ├── catboost.pkl
    └── encoders.pkl
```

---

## Installation

### Local Setup

```bash
git clone https://github.com/lekshanapriya/Flight-Price-Prediction.git
cd Flight-Price-Prediction

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
uvicorn app:app --reload
```

### Docker Deployment

```bash
docker build -t air-fare-predictor .
docker run -p 7860:7860 air-fare-predictor
```

---

## API Endpoints

| Method | Endpoint | Description           |
| ------ | -------- | --------------------- |
| GET    | /health  | Service health check  |
| POST   | /predict | Predict flight fare   |
| GET    | /models  | List available models |

### Example Request

```json
{
  "airline": "IndiGo",
  "source_city": "Delhi",
  "destination_city": "Mumbai",
  "departure_time": "Morning",
  "arrival_time": "Afternoon",
  "stops": 0,
  "class": "Economy"
}
```


## How It Works

1. Input validation and preprocessing
2. Feature engineering (distance, duration, categorical encoding)
3. Model inference using trained Random Forest model
4. Structured JSON response

---



