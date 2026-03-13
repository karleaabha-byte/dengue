import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="India Dengue Risk Dashboard",
    layout="wide"
)

st.title("🦟 Dengue Risk Analysis Dashboard (India)")
st.write("Analysis using CLT and Lyapunov Stability")

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("clean_dengue_india_regions2.csv")

# ensure correct types
df["Year"] = df["Year"].astype(int)
df["Cases"] = df["Cases"].astype(int)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------

st.sidebar.header("Controls")

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
    "Select State / UT",
    regions
)

# filter data
data = df[df["Region"] == region]

# -----------------------------
# BASIC STATISTICS
# -----------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
var_cases = data["Cases"].var()

n = len(data)

# CLT confidence interval
se = std_cases / np.sqrt(n)

ci_low = mean_cases - 1.96 * se
ci_high = mean_cases + 1.96 * se

# -----------------------------
# LYAPUNOV STABILITY
# -----------------------------

data = data.sort_values("Year")

# Lyapunov candidate
data["V"] = data["Cases"]**2

# derivative approximation
data["dV"] = data["V"].diff()

stability_score = data["dV"].mean()

# -----------------------------
# RISK CLASSIFICATION
# -----------------------------

if stability_score < 0:
    risk = "Stable"
elif stability_score < 100000:
    risk = "Moderate Risk"
else:
    risk = "High Outbreak Risk"

# -----------------------------
# DISPLAY METRICS
# -----------------------------

st.subheader(f"Region: {region}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Cases", round(mean_cases,2))
col2.metric("Variance", round(var_cases,2))
col3.metric("CLT CI Lower", round(ci_low,2))
col4.metric("CLT CI Upper", round(ci_high,2))

st.write(f"**Lyapunov Stability Status:** {risk}")

# -----------------------------
# TREND GRAPH
# -----------------------------

st.subheader("Dengue Cases Trend")

fig = px.line(
    data,
    x="Year",
    y="Cases",
    markers=True,
    title=f"Dengue Cases in {region}"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# LYAPUNOV FUNCTION GRAPH
# -----------------------------

st.subheader("Lyapunov Function V(x) = Cases²")

fig2 = px.line(
    data,
    x="Year",
    y="V",
    markers=True,
    title="Lyapunov Function Trend"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# HISTOGRAM (CLT visualization)
# -----------------------------

st.subheader("Distribution of Dengue Cases")

fig3 = px.histogram(
    data,
    x="Cases",
    nbins=15,
    title="Cases Distribution"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# TABLE VIEW
# -----------------------------

st.subheader("Dataset Preview")

st.dataframe(data)

# -----------------------------
# NATIONAL COMPARISON
# -----------------------------

st.subheader("Top 10 High Dengue Regions")

state_mean = df.groupby("Region")["Cases"].mean().reset_index()

top_states = state_mean.sort_values(
    "Cases",
    ascending=False
).head(10)

fig4 = px.bar(
    top_states,
    x="Region",
    y="Cases",
    title="Average Dengue Cases by Region"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------

st.write("---")
st.write("Data Science Project: Dengue Risk Modeling using CLT and Lyapunov Stability")
