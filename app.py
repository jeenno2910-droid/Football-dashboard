import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import poisson
import streamlit as st

st.set_page_config(
    page_title="Football Match Predictor Dashboard", layout="wide"
)

st.title("⚽ AI & Statistical Football Match Predictor Dashboard")
st.markdown(
    "แดชบอร์ดจำลองความน่าจะเป็นผลการแข่งขันฟุตบอลโดยใช้โมเดล **Poisson Distribution** และสถิติอัตราการทำประตู"
)

st.sidebar.header("⚙️ ป้อนสถิติทีมคู่แข่งขัน")

st.sidebar.subheader("🏠 ทีมเหย้า (Home Team)")
team_a = st.sidebar.text_input("ชื่อทีมเหย้า", "Arsenal")
lambda_a = st.sidebar.number_input(
    "ค่าเฉลี่ยประตูยิงได้ (Goals For)", min_value=0.1, max_value=5.0, value=1.8
)
def_a = st.sidebar.number_input(
    "ค่าเฉลี่ยประตูเสีย (Goals Against)", min_value=0.1, max_value=5.0, value=0.8
)

st.sidebar.markdown("---")
st.sidebar.subheader("✈️ ทีมเยือน (Away Team)")
team_b = st.sidebar.text_input("ชื่อทีมเยือน", "Chelsea")
lambda_b = st.sidebar.number_input(
    "ค่าเฉลี่ยประตูยิงได้ (Goals For)", min_value=0.1, max_value=5.0, value=1.4
)
def_b = st.sidebar.number_input(
    "ค่าเฉลี่ยประตูเสีย (Goals Against)", min_value=0.1, max_value=5.0, value=1.1
)

league_avg = 1.35

exp_goals_a = (lambda_a * def_b) / league_avg
exp_goals_b = (lambda_b * def_a) / league_avg

st.subheader(f"📊 ผลวิเคราะห์คาดการณ์: {team_a} vs {team_b}")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label=f"คาดการณ์ประตู {team_a} (xG)", value=f"{exp_goals_a:.2f} ประตู"
    )
with col2:
    st.metric(
        label=f"คาดการณ์ประตู {team_b} (xG)", value=f"{exp_goals_b:.2f} ประตู"
    )

max_goals = 5
prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
for i in range(max_goals + 1):
    for j in range(max_goals + 1):
        prob_matrix[i, j] = poisson.pmf(i, exp_goals_a) * poisson.pmf(
            j, exp_goals_b
        )

home_win = np.sum(np.tril(prob_matrix, -1))
draw = np.sum(np.diag(prob_matrix))
away_win = np.sum(np.triu(prob_matrix, 1))

st.markdown("### 🏆 สรุปเปอร์เซ็นต์ความเป็นไปได้ (Win Probabilities)")
res_cols = st.columns(3)
res_cols[0].metric(
    label=f"{team_a} ชนะ", value=f"{home_win*100:.1f}%", delta="โอกาสคว้าชัย"
)
res_cols[1].metric(
    label="ผลเสมอ (Draw)", value=f"{draw*100:.1f}%", delta="แชร์แต้ม"
)
res_cols[2].metric(
    label=f"{team_b} ชนะ", value=f"{away_win*100:.1f}%", delta="โอกาสทีมเยือน"
)

st.markdown("### 🎯 ตารางเมทริกซ์ความน่าจะเป็นสกอร์แต่ละแบบ (%)")
df_prob = pd.DataFrame(
    prob_matrix * 100,
    index=[f"{team_a} {i}" for i in range(max_goals + 1)],
    columns=[f"{team_b} {j}" for j in range(max_goals + 1)],
)

fig = px.imshow(
    df_prob,
    text_auto=".1f",
    labels=dict(x=f"ประตู {team_b}", y=f"ประตู {team_a}", color="ความน่าจะเป็น (%)"),
    color_continuous_scale="Tealgrn",
)
fig.update_layout(
    title=f"แนวโน้มสกอร์ที่เป็นไปได้มากที่สุดระหว่าง {team_a} และ {team_b}",
    xaxis_title=f"ประตู {team_b}",
    yaxis_title=f"ประตู {team_a}",
)
st.plotly_chart(fig, use_container_width=True)
