import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Dengue Epidemiology Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CUTE STYLE
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color:#FFF7FB;
}

h1 {
    color:#FF5D8F;
    text-align:center;
}

h2,h3 {
    color:#FF7AA2;
}

div[data-testid="metric-container"] {
    background-color:#FFE5EC;
    padding:14px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🦟 Dengue Epidemiology Dashboard")
st.write("🌸 Statistical Outbreak Analysis Across India")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("clean_dengue_india_regions2.csv")

df["Year"] = df["Year"].astype(int)
df["Cases"] = df["Cases"].astype(int)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🧭 Dashboard Controls")

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
    "Select State / UT",
    regions
)

data = df[df["Region"] == region].sort_values("Year").copy()

# --------------------------------------------------
# CLT SECTION
# --------------------------------------------------

st.subheader("📊 Central Limit Theorem Analysis")

st.latex(r"\bar{X} \sim N(\mu, \frac{\sigma}{\sqrt{n}})")

st.latex(r"CI = \bar{X} \pm 1.96 \frac{\sigma}{\sqrt{n}}")

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
var_cases = data["Cases"].var()

n = len(data)

se = std_cases / np.sqrt(n)

ci_low = mean_cases - 1.96 * se
ci_high = mean_cases + 1.96 * se

# --------------------------------------------------
# LYAPUNOV SECTION
# --------------------------------------------------

st.subheader("📉 Lyapunov Stability Analysis")

st.latex(r"V(x) = x^2")

st.latex(r"\Delta V = V(x_{t+1}) - V(x_t)")

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

stability_score = data["dV"].mean()

if stability_score < 0:
    stability = "🟢 Stable"
elif stability_score < 100000:
    stability = "🟡 Moderate Risk"
else:
    stability = "🔴 High Risk"

# --------------------------------------------------
# GROWTH FACTOR SECTION
# --------------------------------------------------

st.subheader("📈 Growth Factor Estimation")

st.latex(r"G_t = \frac{Cases_t}{Cases_{t-1}}")

data["growth_factor"] = data["Cases"] / data["Cases"].shift(1)

data["growth_factor"] = data["growth_factor"].replace([np.inf, -np.inf], np.nan)

valid_growth = data["growth_factor"].dropna()

avg_growth = valid_growth.median()

if np.isnan(avg_growth):
    avg_growth = 1.05

# --------------------------------------------------
# FUTURE PREDICTION MODEL
# --------------------------------------------------

st.subheader("🔮 Future Outbreak Prediction")

st.latex(r"Cases_{t+1} = Cases_t \times G")

last_year = data["Year"].max()
last_cases = data["Cases"].iloc[-1]

future_years = 5
future_data = []

current_cases = last_cases

for i in range(1, future_years + 1):

    current_cases = current_cases * avg_growth

    future_data.append({
        "Year": last_year + i,
        "Cases": round(current_cases)
    })

future_df = pd.DataFrame(future_data)

combined = pd.concat([
    data[["Year", "Cases"]],
    future_df
])

combined["Type"] = ["Actual"] * len(data) + ["Predicted"] * len(future_df)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.subheader(f"📍 Region: {region}")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Average Cases", round(mean_cases, 2))
c2.metric("Variance", round(var_cases, 2))
c3.metric("CLT Lower", round(ci_low, 2))
c4.metric("CLT Upper", round(ci_high, 2))
c5.metric("Growth Factor", round(avg_growth, 3))

st.info(f"Lyapunov Stability Status: **{stability}**")

# --------------------------------------------------
# TREND GRAPH
# --------------------------------------------------

st.subheader("📈 Dengue Cases Trend")

fig1 = px.line(
    data,
    x="Year",
    y="Cases",
    markers=True,
    color_discrete_sequence=["#FF8FAB"]
)

st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------
# LYAPUNOV GRAPH
# --------------------------------------------------

st.subheader("📉 Lyapunov Function Trend")

fig2 = px.line(
    data,
    x="Year",
    y="V",
    markers=True,
    color_discrete_sequence=["#CDB4DB"]
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Cases Distribution")

fig3 = px.histogram(
    data,
    x="Cases",
    nbins=15,
    color_discrete_sequence=["#BDE0FE"]
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# PREDICTION GRAPH
# --------------------------------------------------

st.subheader("🔮 Future Dengue Prediction")

fig4 = px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True,
    color_discrete_sequence=["#FF8FAB", "#A2D2FF"]
)

st.plotly_chart(fig4, use_container_width=True)

# --------------------------------------------------
# TOP REGIONS
# --------------------------------------------------

st.subheader("🏆 Top 10 Regions by Average Cases")

state_mean = df.groupby("Region")["Cases"].mean().reset_index()

top_states = state_mean.sort_values(
    "Cases",
    ascending=False
).head(10)

fig5 = px.bar(
    top_states,
    x="Region",
    y="Cases",
    color="Cases",
    color_continuous_scale="pinkyl"
)

st.plotly_chart(fig5, use_container_width=True)

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.subheader("📋 Data Table")

st.dataframe(data)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("---")
st.caption("🦟 Dengue Risk Modeling | Central Limit Theorem • Lyapunov Stability • Growth Prediction")
