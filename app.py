import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Dengue Stochastic Analysis", layout="wide")

st.title("Dengue Stochastic Analysis Dashboard")

st.write(
"This dashboard analyzes dengue outbreak dynamics using statistical "
"and stochastic modelling techniques."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("clean_dengue_india_regions2.csv")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

df = df.dropna()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
    "Select Region",
    regions
)

data = df[df["Region"] == region].sort_values("Year").copy()

# --------------------------------------------------
# BASIC STATS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
var_cases = data["Cases"].var()

n = len(data)

# --------------------------------------------------
# CENTRAL LIMIT THEOREM
# --------------------------------------------------

st.header("Central Limit Theorem")

st.write("Sample means approach a normal distribution as sample size increases.")

st.latex(r"\bar{X} \sim N(\mu,\frac{\sigma}{\sqrt{n}})")

se = std_cases / np.sqrt(n)

ci_low = mean_cases - 1.96 * se
ci_high = mean_cases + 1.96 * se

c1,c2,c3 = st.columns(3)

c1.metric("Mean Cases", round(mean_cases,2))
c2.metric("Std Deviation", round(std_cases,2))
c3.metric("Standard Error", round(se,2))

st.write("95% Confidence Interval:", round(ci_low,2), "to", round(ci_high,2))

# --------------------------------------------------
# VARIANCE
# --------------------------------------------------

st.header("Variance")

st.write("Variance measures stochastic variability in outbreak intensity.")

st.latex(r"Var(X)=\frac{1}{n}\sum (X_i-\mu)^2")

st.metric("Variance", round(var_cases,2))

# --------------------------------------------------
# COEFFICIENT OF VARIATION
# --------------------------------------------------

st.header("Coefficient of Variation")

st.write("Relative variability of the outbreak compared to the mean.")

st.latex(r"CV=\frac{\sigma}{\mu}")

cv = std_cases / mean_cases if mean_cases != 0 else np.nan

st.metric("Coefficient of Variation", round(cv,3))

# --------------------------------------------------
# GROWTH FACTOR
# --------------------------------------------------

st.header("Growth Factor")

st.write("Measures year-to-year change in dengue cases.")

st.latex(r"G_t=\frac{Cases_t-Cases_{t-1}}{Cases_{t-1}}")

data["growth"] = data["Cases"].pct_change()

growth = data["growth"].replace([np.inf,-np.inf],np.nan).dropna()

avg_growth = growth.median() if len(growth)>0 else 0

st.metric("Median Growth Rate", round(avg_growth,3))

# --------------------------------------------------
# LYAPUNOV STABILITY
# --------------------------------------------------

st.header("Lyapunov Stability")

st.write("Evaluates whether outbreak dynamics converge or diverge.")

st.latex(r"V(x)=x^2")

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

lyapunov_value = data["dV"].mean()

c1,c2 = st.columns(2)

c1.metric("Lyapunov Value", round(lyapunov_value,2))

if lyapunov_value < 0:
    status = "Stable"
elif lyapunov_value < 100000:
    status = "Moderate Risk"
else:
    status = "High Outbreak Risk"

c2.metric("System Stability", status)

# --------------------------------------------------
# OBSERVED TREND GRAPH
# --------------------------------------------------

st.header("Observed Dengue Case Trend")

fig = px.line(
    data,
    x="Year",
    y="Cases",
    markers=True,
    color_discrete_sequence=["#e75480"]
)

fig.update_layout(
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

st.header("Monte Carlo Simulation")

st.write("Simulates possible future outbreak trajectories using stochastic growth.")

last_cases = data["Cases"].iloc[-1]

years = 5
simulations = 100

sim_data = []

for s in range(simulations):

    current = last_cases
    path = []

    for y in range(years):

        g = np.random.normal(avg_growth,0.1)

        current = current * (1 + g)

        path.append(current)

    sim_data.append(path)

sim_df = pd.DataFrame(sim_data).T

fig_sim = go.Figure()

for col in sim_df.columns:

    fig_sim.add_trace(
        go.Scatter(
            y=sim_df[col],
            mode="lines",
            line=dict(color="rgba(231,84,128,0.15)"),
            showlegend=False
        )
    )

fig_sim.update_layout(
    template="plotly_white",
    xaxis_title="Future Years",
    yaxis_title="Simulated Cases"
)

st.plotly_chart(fig_sim, use_container_width=True)

# --------------------------------------------------
# FUTURE PREDICTION
# --------------------------------------------------

st.header("Future Growth Prediction")

st.write("Future cases estimated using multiplicative growth dynamics.")

st.latex(r"Cases_{t+1}=Cases_t(1+G)")

future = []
current = last_cases
last_year = data["Year"].max()

for i in range(1,6):

    current = current * (1 + avg_growth)

    future.append({
        "Year": last_year + i,
        "Cases": current
    })

future_df = pd.DataFrame(future)

combined = pd.concat([data[["Year","Cases"]],future_df])

combined["Type"] = ["Actual"]*len(data) + ["Predicted"]*len(future_df)

fig2 = px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True,
    color_discrete_sequence=["#e75480","#ff9eb5"]
)

fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------

st.header("Case Distribution")

fig3 = px.histogram(
    data,
    x="Cases",
    nbins=15,
    color_discrete_sequence=["#f8a5c2"]
)

fig3.update_layout(template="plotly_white")

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------

st.header("Dataset")

st.dataframe(data)
