"""The rageval app: one program with two features, chosen at the top of the sidebar."""
import streamlit as st

from rageval.ui import run_app

FEATURES = {"Feature 1: Dataset generation": "datagen", "Feature 2: Auto evaluation": "evaluation"}

st.set_page_config(page_title="rageval", layout="wide")
feature = st.sidebar.radio("Feature", list(FEATURES), key="feature")
st.sidebar.divider()
if FEATURES[feature] == "datagen":
    from rageval.datagen.app import PAGES
else:
    from rageval.evaluation.app import PAGES  # only the chosen feature is imported, so the program starts faster
run_app(PAGES, key=f"page|{FEATURES[feature]}")
