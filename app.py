import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Dengue Outbreak Analysis",
    layout="wide"
)

st.title("Dengue Outbreak Stochastic Analysis Dashboard")

st.write(
"This dashboard analyzes dengue outbreak dynamics across Indian regions "
"using statistical and stochastic modeling techniques."
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

data = df[df["Region"] == region].sort_values("Year")

# --------------------------------------------------
# BASIC STATS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
variance = data["Cases"].var()

n = len(data)

# --------------------------------------------------
# CENTRAL LIMIT THEOREM
# --------------------------------------------------

st.header("Central Limit Theorem")

st.write(
"Sample means tend toward a normal distribution as the sample size increases."
)

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

st.write(
"Variance measures how widely dengue case counts fluctuate over time."
)

st.latex(r"Var(X)=\frac{1}{n}\sum (X_i-\mu)^2")

st.metric("Variance", round(variance,2))

# --------------------------------------------------
# COEFFICIENT OF VARIATION
# --------------------------------------------------

st.header("Coefficient of Variation")

st.write(
"Coefficient of variation measures relative variability compared to the mean."
)

st.latex(r"CV=\frac{\sigma}{\mu}")

cv = std_cases / mean_cases if mean_cases != 0 else np.nan

st.metric("CV", round(cv,3))

# --------------------------------------------------
# GROWTH FACTOR
# --------------------------------------------------

st.header("Growth Factor")

st.write(
"Growth factor measures year-to-year percentage change in cases."
)

st.latex(r"G_t=\frac{Cases_t-Cases_{t-1}}{Cases_{t-1}}")

data["growth"] = data["Cases"].pct_change()

growth = data["growth"].replace([np.inf,-np.inf],np.nan).dropna()

avg_growth = growth.median() if len(growth)>0 else 0

st.metric("Median Growth Rate", round(avg_growth,3))

# --------------------------------------------------
# LYAPUNOV STABILITY
# --------------------------------------------------

st.header("Lyapunov Stability")

st.write(
"Lyapunov analysis evaluates whether outbreak dynamics are stable or diverging."
)

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
# TREND GRAPH
# --------------------------------------------------

st.header("Observed Dengue Trend")

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

st.header("Monte Carlo Outbreak Simulation")

st.write(
"Monte Carlo simulation generates multiple possible future outbreak paths "
"using stochastic growth dynamics."
)

st.latex(r"Cases_{t+1}=Cases_t(1+G+\epsilon)")

simulations = 200
future_years = 5

last_cases = data["Cases"].iloc[-1]
last_year = int(data["Year"].max())

years = list(range(last_year+1,last_year+future_years+1))

paths = []

for s in range(simulations):

    current = last_cases
    path = []

    for y in years:

        noise = np.random.normal(0,0.05)

        current = current * (1 + avg_growth + noise)

        path.append(current)

    paths.append(path)

paths = np.array(paths)

mean_path = paths.mean(axis=0)
upper = np.percentile(paths,95,axis=0)
lower = np.percentile(paths,5,axis=0)

fig_sim = go.Figure()

fig_sim.add_trace(
    go.Scatter(
        x=years,
        y=upper,
        line=dict(width=0),
        showlegend=False
    )
)

fig_sim.add_trace(
    go.Scatter(
        x=years,
        y=lower,
        fill="tonexty",
        fillcolor="rgba(231,84,128,0.2)",
        line=dict(width=0),
        name="Uncertainty Range"
    )
)

fig_sim.add_trace(
    go.Scatter(
        x=years,
        y=mean_path,
        line=dict(color="#e75480",width=4),
        mode="lines+markers",
        name="Expected Cases"
    )
)

fig_sim.update_layout(
    template="plotly_white",
    xaxis_title="Year",
    yaxis_title="Predicted Cases"
)

st.plotly_chart(fig_sim,use_container_width=True)

# --------------------------------------------------
# FUTURE PREDICTION
# --------------------------------------------------

st.header("Future Growth Prediction")

st.write(
"Future cases estimated using multiplicative growth dynamics."
)

st.latex(r"Cases_{t+1}=Cases_t(1+G)")

future = []

current = last_cases

for i in range(1,6):

    current = current*(1+avg_growth)

    future.append({
        "Year": last_year+i,
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

st.plotly_chart(fig2,use_container_width=True)

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

st.plotly_chart(fig3,use_container_width=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------

st.header("Dataset")

st.dataframe(data)
