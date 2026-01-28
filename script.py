import pandas as pd
import streamlit as st
import time
import re
from datetime import datetime

SHEET_ID = "1VF9956Fx_HVbvRV2XSwRhFkQihL1Ke6B2GXkc4xDXKo"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

TOPICS = {
    0: "Cryptography",
    1: "Authentication",
    2: "User Privacy",
    3: "Network Security"
}

def parse_priorities(val):
    if not isinstance(val, str):
        return None
    match = re.search(r"\[(\d+,\d+,\d+,\d+)\]", val)
    if not match:
        return None
    return list(map(int, match.group(1).split(",")))

st.set_page_config(layout="wide")
st.title("Topic Priority Distribution")

while True:
    df = pd.read_csv(URL)
    df["priorities"] = df["Automatically generated!"].apply(parse_priorities)
    df = df.dropna(subset=["priorities"])

    rows = []
    for idx, name in TOPICS.items():
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for arr in df["priorities"]:
            counts[arr[idx]] += 1
        rows.append(counts)

    chart_df = pd.DataFrame(rows, index=TOPICS.values())
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Priority 1")
        st.bar_chart(chart_df[1])

    with col2:
        st.subheader("Priority 2")
        st.bar_chart(chart_df[2])

    with col3:
        st.subheader("Priority 3")
        st.bar_chart(chart_df[3])

    with col4:
        st.subheader("Priority 4")
        st.bar_chart(chart_df[4])

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(10)
    st.rerun()