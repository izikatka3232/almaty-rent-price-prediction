import sqlite3
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from catboost import CatBoostRegressor

import os
os.makedirs("models", exist_ok=True)

# ─── 1. ЗАГРУЗКА ───────────────────────────────────────────
conn = sqlite3.connect("../parser/db.sqlite")

df = pd.read_sql_query("""
    SELECT f.id, f.room, f.square, f.city,
           f.lat, f.lon, f.star, f.focus, p.price
    FROM flats f
    LEFT JOIN prices p ON f.id = p.flat_id
""", conn)

conn.close()
print(f"Загружено записей: {len(df)}")

# ─── 2. ОЧИСТКА ────────────────────────────────────────────
df = df.dropna(subset=["price"])
df = df[(df["price"] > 0) & (df["square"] > 0)]

q_low  = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]

print(f"После очистки: {len(df)}")

feat = ["room", "square", "lat", "lon", "star", "focus"]
target   = "price"

X = df[feat]
y = df[target]

def make_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])

models = {
    "LinearRegression": make_pipe(LinearRegression()),
    "Ridge":            make_pipe(Ridge(alpha=10)),
    "Lasso":            make_pipe(Lasso(alpha=10)),
    "RandomForest":     make_pipe(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    "XGBoost":          make_pipe(xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42, verbosity=0)),
    "CatBoost":         make_pipe(CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0)),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

print(f"\n{'Model':<22} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
print("─" * 58)

for name, pipe in models.items():
    rmse = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    mae  = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error",     n_jobs=-1)
    r2   =  cross_val_score(pipe, X, y, cv=kf, scoring="r2",                          n_jobs=-1)

    results.append({
        "model":    name,
        "RMSE":     rmse.mean(),
        "RMSE_std": rmse.std(),
        "MAE":      mae.mean(),
        "R2":       r2.mean(),
    })
    print(f"{name:<22} {rmse.mean():>12,.0f} {mae.mean():>12,.0f} {r2.mean():>8.3f}")

res_df = pd.DataFrame(results).sort_values("RMSE")

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle("Сравнение моделей — 5-fold CV", fontsize=14)

# for ax, metric, color in zip(axes, ["RMSE", "MAE", "R2"], ["#E8593C", "#3B8BD4", "#1D9E75"]):
#     bars = ax.barh(res_df["model"], res_df[metric], color=color, alpha=0.85)
#     for bar, val in zip(bars, res_df[metric]):
#         label = f"{val:,.0f}" if metric != "R2" else f"{val:.3f}"
#         ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
#                 label, va="center", fontsize=8)
#     ax.set_title(metric)
#     ax.invert_yaxis()

# plt.tight_layout()
# plt.savefig("models/model_comparison.png", dpi=150, bbox_inches="tight")
# plt.show()

best_name = res_df.iloc[0]["model"]
print(f"\nЛучшая модель: {best_name}")

best_pipe = models[best_name]
best_pipe.fit(X, y)

# if hasattr(best_pipe["model"], "feature_importances_"):
#     fi = pd.Series(best_pipe["model"].feature_importances_, index=feat).sort_values()
#     fi.plot(kind="barh", figsize=(7, 4), color="#3B8BD4",
#             title=f"Feature importance — {best_name}")
#     plt.tight_layout()
#     plt.savefig("models/feature_importance.png", dpi=150, bbox_inches="tight")
#     plt.show()

# ─── 8. СОХРАНЯЕМ МОДЕЛЬ И МЕТАДАННЫЕ ──────────────────────
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

best_row = res_df[res_df["model"] == best_name].iloc[0]

meta = {
    "model_name": best_name,
    "features": feat,
    "metrics": {
        "RMSE": round(best_row["RMSE"], 2),
        "MAE":  round(best_row["MAE"],  2),
        "R2":   round(best_row["R2"],   4),
    },
    "feature_ranges": {
        col: {
            "min":    float(df[col].min()),
            "max":    float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in feat
    },
    "price_range": {
        "min":    float(df[target].min()),
        "max":    float(df[target].max()),
        "median": float(df[target].median()),
    },
    "cv_results": res_df.to_dict(orient="records"),
}

with open("models/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\nГотово! Запусти приложение:")
print("  streamlit run app.py")