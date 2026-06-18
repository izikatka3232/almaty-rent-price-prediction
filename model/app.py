
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Оценка аренды — Алматы",
    page_icon="🏠",
    layout="wide",
)

MODELS_DIR = Path("models")

ALMATY_CENTER = {"lat": 43.238949, "lon": 76.889709}


@st.cache_resource
def load_model():
    path = MODELS_DIR / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    path = MODELS_DIR / "meta.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta  = load_meta()

st.title("🏠 Оценка стоимости аренды — Алматы")
st.caption("Модель обучена на данных krisha.kz · Предсказание в тенге (₸)")

if model is None or meta is None:
    st.error(
        "Модель не найдена. Сначала запусти обучение:\n\n"
        "```bash\npython train.py\n```"
    )
    st.stop()


with st.sidebar:
    st.header("📊 Информация о модели")
    st.write(f"**Модель:** {meta['model_name']}")

    m = meta["metrics"]
    st.metric("R²",   f"{m['R2']:.3f}")
    st.metric("RMSE", f"{m['RMSE']:,.0f} ₸")
    st.metric("MAE",  f"{m['MAE']:,.0f} ₸")

    st.divider()
    st.caption(
        "R² показывает, какую долю дисперсии цены объясняет модель "
        "(1.0 = идеально). RMSE и MAE — средние ошибки в тенге."
    )

    st.divider()
    st.header("📈 Сравнение моделей")
    cv_df = pd.DataFrame(meta["cv_results"]).sort_values("RMSE")
    fig_cv = px.bar(
        cv_df, x="RMSE", y="model", orientation="h",
        color="R2", color_continuous_scale="teal",
        labels={"model": "", "RMSE": "RMSE (₸)", "R2": "R²"},
        height=280,
    )
    fig_cv.update_layout(margin=dict(l=0, r=10, t=10, b=0),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_cv, use_container_width=True)


col_input, col_result = st.columns([1, 1], gap="large")

fr = meta["feature_ranges"]

with col_input:
    st.subheader("Параметры квартиры")

    rooms = st.selectbox(
        "Количество комнат",
        options=[1, 2, 3, 4, 5],
        index=1,
    )

    square = st.slider(
        "Площадь, м²",
        min_value=int(fr["square"]["min"]),
        max_value=min(int(fr["square"]["max"]), 300),
        value=int(fr["square"]["median"]),
        step=1,
    )

    star_min = int(fr["star"]["min"])
    star_max = int(fr["star"]["max"])
    star_val = int(fr["star"]["median"])
    if star_min < star_max:
        star = st.slider(
            "Рейтинг объявления (star)",
            min_value=star_min, max_value=star_max, value=star_val,
            help="Внутренний рейтинг krisha.kz",
        )
    else:
        star = st.number_input(
            "Рейтинг объявления (star)",
            value=float(star_val),
            help="Внутренний рейтинг krisha.kz (в данных одно уникальное значение)",
        )

    focus_min = float(fr["focus"]["min"])
    focus_max = float(fr["focus"]["max"])
    focus_val = float(fr["focus"]["median"])
    if focus_min < focus_max:
        focus = st.slider(
            "Focus-score объявления",
            min_value=focus_min, max_value=focus_max, value=focus_val,
            step=max(round((focus_max - focus_min) / 100, 2), 0.01),
            help="Показатель качества объявления на krisha.kz",
        )
    else:
        focus = st.number_input(
            "Focus-score объявления",
            value=focus_val,
            help="Focus-score (в данных одно уникальное значение)",
        )

    st.markdown("##### 📍 Местоположение")
    st.caption("Перетащи маркер или введи координаты вручную")

    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input(
            "Широта (lat)",
            value=ALMATY_CENTER["lat"],
            format="%.6f", step=0.001,
        )
    with c2:
        lon = st.number_input(
            "Долгота (lon)",
            value=ALMATY_CENTER["lon"],
            format="%.6f", step=0.001,
        )

    predict_btn = st.button("🔍 Оценить стоимость", use_container_width=True,
                            type="primary")


with col_result:
    st.subheader("Результат")

    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=12, use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "room":   rooms,
            "square": square,
            "lat":    lat,
            "lon":    lon,
            "star":   star,
            "focus":  focus,
        }])

        prediction = model.predict(input_df)[0]
        price_per_sqm = prediction / square

        low  = prediction * 0.85
        high = prediction * 1.15

        st.success(f"### {prediction:,.0f} ₸ / мес")

        c1, c2, c3 = st.columns(3)
        c1.metric("Нижняя оценка", f"{low:,.0f} ₸")
        c2.metric("Медианная",     f"{prediction:,.0f} ₸")
        c3.metric("Верхняя оценка", f"{high:,.0f} ₸")

        st.metric("Цена за м²", f"{price_per_sqm:,.0f} ₸/м²")

        price_min = meta["price_range"]["min"]
        price_max = meta["price_range"]["max"]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " ₸", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [price_min, price_max]},
                "bar":  {"color": "#1D9E75"},
                "steps": [
                    {"range": [price_min,
                               meta["price_range"]["median"] * 0.7],
                     "color": "#e8f5e9"},
                    {"range": [meta["price_range"]["median"] * 0.7,
                               meta["price_range"]["median"] * 1.3],
                     "color": "#c8e6c9"},
                    {"range": [meta["price_range"]["median"] * 1.3,
                               price_max],
                     "color": "#a5d6a7"},
                ],
                "threshold": {
                    "line": {"color": "#E8593C", "width": 3},
                    "thickness": 0.75,
                    "value": meta["price_range"]["median"],
                },
            },
            title={"text": "Относительно рынка"},
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"🔴 Красная отметка = медиана рынка "
            f"({meta['price_range']['median']:,.0f} ₸)"
        )
    else:
        st.info("Задай параметры слева и нажми **«Оценить стоимость»**")


st.divider()
st.caption(
    "Данные: krisha.kz · Модель: ML pipeline на sklearn/XGBoost/CatBoost · "
    "Ошибка модели ≈ ±MAE ₸"
)