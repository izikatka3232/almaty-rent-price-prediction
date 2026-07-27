[![RU](https://img.shields.io/badge/README-RU-red.svg)](README.ru.md)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn_|_XGBoost_|_CatBoost-orange)
![App](https://img.shields.io/badge/app-Streamlit-FF4B4B)

# Almaty Rental Price Prediction

An end-to-end machine learning project that predicts monthly apartment rental
prices in Kazakhstan from listing data, served through an interactive Streamlit app.

The pipeline goes from raw scraped listings → cleaning → model selection
(6 regressors, 5-fold CV) → a deployed web app that returns a price estimate,
a per-m² breakdown and a market-position gauge.

> **Data source.** Listings are collected with an external parser,
> [krisha.kz by andprov](https://github.com/andprov/krisha.kz) (MIT License).
> My work covers the **ML pipeline, modeling and the application layer** built
> on top of the collected data.

---

## Demo

The app takes apartment parameters (rooms, area, location) and returns:

- a point price estimate in KZT with a ±15% range,
- price per m²,
- a gauge showing where the estimate sits relative to the market median,
- a map of the selected location.

![Model comparison](reports/model_comparison.png)
![Feature importance](reports/feature_importance.png)

---

## Results

Six regression models were compared with 5-fold cross-validation on the cleaned
dataset. **XGBoost** performed best.

| Model            | RMSE (₸)  | MAE (₸)   | R²     |
|------------------|----------:|----------:|-------:|
| **XGBoost**      | **63 741**| **41 148**| **0.778** |
| RandomForest     | 66 272    | 43 196    | 0.760  |
| CatBoost         | 66 387    | 43 956    | 0.759  |
| Ridge            | 94 367    | 63 406    | 0.510  |
| Lasso            | 94 384    | 63 419    | 0.510  |
| LinearRegression | 94 396    | 63 419    | 0.510  |

The gap between the tree ensembles (R² ≈ 0.76–0.78) and the linear models
(R² ≈ 0.51) shows the price surface is strongly non-linear — location and area
interact in ways linear models can't capture.

---

## How it works

```
parser (external) ──> SQLite ──> cleaning ──> 5-fold CV model selection ──> best_model.pkl + meta.json ──> Streamlit app
```

**Pipeline (`train.py`)**
- Loads `flats` joined with `prices` from the parser's SQLite DB.
- Cleans data: drops missing/zero prices, clips the 1st/99th price percentiles
  to remove outliers.
- Features: `room`, `square`, `lat`, `lon`, `star`, `focus`.
- Each model runs in a `Pipeline` (median imputation → standard scaling → model).
- Selects the best model by cross-validated RMSE, refits on all data, and saves
  `best_model.pkl` plus a `meta.json` with metrics, feature ranges and CV results.

**App (`app.py`)**
- Loads the saved model and metadata.
- Interactive inputs for apartment parameters and map location.
- Returns the prediction, a confidence range, price per m², and a Plotly gauge
  positioning the estimate against the market median.

---

## Tech stack

- **Language:** Python
- **ML:** scikit-learn, XGBoost, CatBoost
- **Data:** pandas, NumPy, SQLite
- **App / viz:** Streamlit, Plotly, Matplotlib

---

## Installation & running

```bash
# 1. clone
git clone https://github.com/m1maka-creator/<repo-name>.git
cd <repo-name>

# 2. environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. collect data with the parser (see parser/README.md), then train
python train.py

# 4. run the app
streamlit run app.py
```

A pre-trained `models/best_model.pkl` is included, so you can launch the app
directly without re-collecting data.

---

## Project structure

```
.
├── README.md / README.ru.md
├── train.py              # data load, cleaning, model selection, save artifacts
├── app.py                # Streamlit web app
├── models/
│   ├── best_model.pkl    # trained pipeline
│   └── meta.json         # metrics, feature ranges, CV results
├── reports/              # model_comparison.png, feature_importance.png
└── parser/               # external scraper (andprov, MIT) — see credits
```

---

## Limitations & next steps

Being explicit about the current state:

- **`star` and `focus` are constant** (0 for every row) in the current dump, so
  they carry no signal — effectively the model runs on rooms, area and
  coordinates. Removing them is the first cleanup step.
- **Geographic spread.** The dump contains listings beyond Almaty (some
  coordinates fall in other regions), so the model is broader than the app title
  suggests; filtering strictly to Almaty would tighten estimates.
- **Planned improvements:** add district/neighbourhood and floor features,
  wrap the model in a **FastAPI** endpoint, containerize with **Docker**, and add
  experiment tracking (**MLflow**).

---

## Credits

- Data collection: [krisha.kz parser by andprov](https://github.com/andprov/krisha.kz), MIT License.
