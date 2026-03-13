import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Dengue Stochastic Analysis Dashboard",
    layout="wide"
)

st.title("Dengue Outbreak Stochastic Analysis Dashboard")

st.write(
"This dashboard analyzes dengue case dynamics across Indian regions "
"using statistical inference, stochastic variability measures, "
"and predictive growth modelling."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("clean_dengue_india_regions2.csv")

df["Year"] = df["Year"].astype(int)
df["Cases"] = df["Cases"].astype(int)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Controls")

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
    "Select State / UT",
    regions
)

data = df[df["Region"] == region].sort_values("Year").copy()

# --------------------------------------------------
# BASIC METRICS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
var_cases = data["Cases"].var()
n = len(data)

# --------------------------------------------------
# SECTION 1
# CENTRAL LIMIT THEOREM
# --------------------------------------------------

st.header("Central Limit Theorem")

st.write(
"The Central Limit Theorem states that the distribution of sample means "
"approaches a normal distribution as the sample size increases."
)

st.latex(r"\bar{X} \sim N(\mu,\frac{\sigma}{\sqrt{n}})")

se = std_cases / np.sqrt(n)

ci_low = mean_cases - 1.96 * se
ci_high = mean_cases + 1.96 * se

c1,c2 = st.columns(2)

c1.metric("Mean Cases", round(mean_cases,2))
c2.metric("Standard Error", round(se,2))

st.write("95% Confidence Interval:", round(ci_low,2), "to", round(ci_high,2))

# --------------------------------------------------
# SECTION 2
# VARIANCE
# --------------------------------------------------

st.header("Variance Analysis")

st.write(
"Variance measures the dispersion of dengue cases, representing "
"the level of stochastic variability in the outbreak process."
)

st.latex(r"Var(X) = \frac{1}{n}\sum (X_i - \mu)^2")

st.metric("Variance", round(var_cases,2))

# --------------------------------------------------
# SECTION 3
# COEFFICIENT OF VARIATION
# --------------------------------------------------

st.header("Coefficient of Variation")

st.write(
"The coefficient of variation measures relative variability "
"and indicates outbreak instability relative to the mean."
)

st.latex(r"CV = \frac{\sigma}{\mu}")

cv = std_cases / mean_cases

st.metric("Coefficient of Variation", round(cv,3))

# --------------------------------------------------
# SECTION 4
# GROWTH FACTOR
# --------------------------------------------------

st.header("Growth Factor Estimation")

st.write(
"The growth factor measures how dengue cases change from one year "
"to the next, capturing epidemic expansion or decline."
)

st.latex(r"G_t = \frac{Cases_t}{Cases_{t-1}}")

data["growth"] = data["Cases"] / data["Cases"].shift(1)

data["growth"] = data["growth"].replace([np.inf,-np.inf],np.nan)

growth = data["growth"].dropna()

avg_growth = growth.median()

if np.isnan(avg_growth):
    avg_growth = 1.05

st.metric("Median Growth Factor", round(avg_growth,3))

# --------------------------------------------------
# SECTION 5
# LYAPUNOV STABILITY
# --------------------------------------------------

st.header("Lyapunov Stability Analysis")

st.write(
"Lyapunov functions help determine whether the outbreak dynamics "
"are stabilizing or diverging over time."
)

st.latex(r"V(x) = x^2")

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

score = data["dV"].mean()

if score < 0:
    status = "Stable"
elif score < 100000:
    status = "Moderate Risk"
else:
    status = "High Outbreak Risk"

st.metric("Stability Status", status)

# --------------------------------------------------
# SECTION 6
# MONTE CARLO SIMULATION
# --------------------------------------------------

st.header("Monte Carlo Outbreak Simulation")

st.write(
"Monte Carlo simulation estimates possible future outbreak trajectories "
"by sampling stochastic growth variations."
)

simulations = []

last_cases = data["Cases"].iloc[-1]

for i in range(1000):
    
    g = np.random.normal(avg_growth,0.1)
    
    future = last_cases
    
    for j in range(5):
        future = future * g
        
    simulations.append(future)

sim_mean = np.mean(simulations)

st.metric("Expected Cases in 5 Years", round(sim_mean))

# --------------------------------------------------
# SECTION 7
# FUTURE PREDICTION
# --------------------------------------------------

st.header("Future Growth Prediction")

st.write(
"Future dengue cases are projected using multiplicative growth dynamics."
)

st.latex(r"Cases_{t+1} = Cases_t \times G")

future = []
current = last_cases
last_year = data["Year"].max()

for i in range(1,6):
    
    current = current * avg_growth
    
    future.append({
        "Year": last_year + i,
        "Cases": current
    })

future_df = pd.DataFrame(future)

combined = pd.concat([data[["Year","Cases"]],future_df])

combined["Type"] = ["Actual"]*len(data) + ["Predicted"]*len(future_df)

fig = px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True
)

st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# CASE TREND
# --------------------------------------------------

st.header("Observed Dengue Case Trend")

fig2 = px.line(
    data,
    x="Year",
    y="Cases",
    markers=True
)

st.plotly_chart(fig2,use_container_width=True)

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------

st.header("Case Distribution")

fig3 = px.histogram(
    data,
    x="Cases",
    nbins=15
)

st.plotly_chart(fig3,use_container_width=True)

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.header("Dataset")

st.dataframe(data)
