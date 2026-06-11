from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from generate_dataset import extract_url_features
from train_model import FEATURE_COLUMNS

DATA_PATH = Path("data/phishing_sites_synthetic.csv")
MODEL_PATH = Path("models/phishing_detector.joblib")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide")

CUSTOM_CSS = """
<style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {color: #5f6b7a; font-size: 1.05rem; margin-bottom: 1.3rem;}
    .risk-card {padding: 1.2rem; border-radius: 18px; border: 1px solid #e5e7eb; background: #ffffff; box-shadow: 0 8px 24px rgba(15,23,42,.06);}
    .good {color: #047857; font-weight: 800;}
    .bad {color: #b91c1c; font-weight: 800;}
    .metric-note {font-size: .9rem; color: #64748b;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="main-title">🛡️ CyberShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-система для обнаружения фишинговых сайтов по признакам URL и веб-страницы</div>', unsafe_allow_html=True)


def ensure_artifacts() -> None:
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        subprocess.run([sys.executable, "train_model.py"], check=True)


@st.cache_resource
def load_model():
    ensure_artifacts()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    ensure_artifacts()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    ensure_artifacts()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


model = load_model()
df = load_data()
metrics = load_metrics()

with st.sidebar:
    st.header("Навигация")
    page = st.radio("Раздел", ["Проверка сайта", "Анализ датасета", "О модели", "О проекте"])
    st.divider()
    st.caption("Классы: 0 — легитимный сайт, 1 — фишинг")
    st.caption("Датасет синтетический, создан для учебного проекта.")

if page == "Проверка сайта":
    col_left, col_right = st.columns([1.05, 0.95])

    with col_left:
        st.subheader("Введите сайт для проверки")
        url = st.text_input(
            "URL",
            value="http://sber-login-secure.top/verify/account?session=932811",
            help="URL-признаки извлекаются автоматически: длина, цифры, дефисы, подозрительные слова и т.д.",
        )
        st.markdown("**Дополнительные признаки страницы**  ")
        c1, c2 = st.columns(2)
        with c1:
            domain_age_days = st.slider("Возраст домена, дней", 1, 5000, 45)
            forms_count = st.slider("Количество форм на странице", 0, 8, 2)
            ssl_valid = st.selectbox("SSL-сертификат валиден?", ["Да", "Нет"], index=1)
        with c2:
            external_links_ratio = st.slider("Доля внешних ссылок", 0.0, 1.0, 0.72, 0.01)
            popup_count = st.slider("Количество pop-up окон", 0, 8, 1)
            redirect_count = st.slider("Количество редиректов", 0, 8, 2)

        check = st.button("Проверить", type="primary", use_container_width=True)

    with col_right:
        if check:
            features = extract_url_features(url)
            features.update({
                "domain_age_days": domain_age_days,
                "external_links_ratio": external_links_ratio,
                "forms_count": forms_count,
                "popup_count": popup_count,
                "ssl_valid": int(ssl_valid == "Да"),
                "redirect_count": redirect_count,
            })
            X = pd.DataFrame([features])[FEATURE_COLUMNS]
            risk = float(model.predict_proba(X)[0, 1])
            label = int(risk >= 0.5)

            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            if label:
                st.markdown(f"### <span class='bad'>Вероятно фишинг</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### <span class='good'>Похоже на легитимный сайт</span>", unsafe_allow_html=True)
            st.metric("Риск фишинга", f"{risk * 100:.1f}%")
            st.progress(min(max(risk, 0), 1))
            st.caption("Это учебная ML-оценка, а не полноценный антивирусный вердикт.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("Признаки, отправленные в модель:")
            st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
        else:
            st.info("Заполните признаки и нажмите «Проверить».")

elif page == "Анализ датасета":
    st.subheader("EDA: быстрый анализ синтетического датасета")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Строк", len(df))
    m2.metric("Признаков", len(FEATURE_COLUMNS))
    m3.metric("Фишинг", f"{df['label'].mean() * 100:.1f}%")
    m4.metric("Средняя длина URL", f"{df['url_length'].mean():.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Распределение классов")
        st.bar_chart(df["label"].map({0: "legit", 1: "phishing"}).value_counts())
    with c2:
        st.write("Средние значения признаков по классам")
        selected = st.selectbox("Признак", FEATURE_COLUMNS, index=0)
        st.bar_chart(df.groupby("label")[selected].mean().rename(index={0: "legit", 1: "phishing"}))

    st.write("Корреляции признаков с меткой phishing")
    corr = df[FEATURE_COLUMNS + ["label"]].corr(numeric_only=True)["label"].drop("label").sort_values(key=abs, ascending=False)
    st.bar_chart(corr)

    with st.expander("Показать первые строки датасета"):
        st.dataframe(df.head(30), use_container_width=True)

elif page == "О модели":
    st.subheader("Качество модели")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metrics["accuracy"])
    c2.metric("F1-score", metrics["f1"])
    c3.metric("ROC-AUC", metrics["roc_auc"])
    st.caption("Метрики считаются на отложенной выборке train/test split.")

    cm = pd.DataFrame(metrics["confusion_matrix"], index=["real legit", "real phishing"], columns=["pred legit", "pred phishing"])
    st.write("Матрица ошибок")
    st.dataframe(cm, use_container_width=True)

    st.write("Используемые признаки")
    st.code("\n".join(FEATURE_COLUMNS), language="text")

    if st.button("Переобучить модель"):
        subprocess.run([sys.executable, "train_model.py"], check=True)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Модель переобучена. Обновите страницу, чтобы увидеть новые артефакты.")

else:
    st.subheader("Идея проекта")
    st.write(
        "CyberShield AI — учебный проект по машинному обучению: модель RandomForest анализирует URL "
        "и признаки веб-страницы, чтобы оценить вероятность фишинга. Проект подходит для GitHub: "
        "есть генератор датасета, обучение, метрики, сохранённая модель и Streamlit-интерфейс."
    )
    st.markdown(
        """
        **Что можно рассказать на защите:**
        - датасет синтетический, но признаки имитируют реальные факторы риска;
        - задача — бинарная классификация;
        - модель обучается на числовых признаках;
        - интерфейс позволяет менять признаки и сразу видеть вероятность фишинга;
        - есть EDA-раздел с распределениями и корреляциями.
        """
    )
