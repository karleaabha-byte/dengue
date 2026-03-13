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

# --------------------------------------------------
# PROFESSIONAL PINK THEME
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color:#faf7fb;
}

.analysis-card {
    background:white;
    padding:25px;
    border-radius:15px;
    border-left:6px solid #e75480;
    box-shadow:0px 4px 12px rgba(0,0,0,0.05);
    margin-bottom:25px;
}

.section-title {
    font-size:24px;
    font-weight:600;
    color:#c2185b;
}

.explain {
    color:#555;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
"""
<h1 style='text-align:center;color:#c2185b'>
Dengue Outbreak Stochastic Analysis Dashboard
</h1>
""",
unsafe_allow_html=True
)

st.write(
"Statistical and stochastic modelling of dengue outbreaks across Indian regions."
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
# BASIC STATS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
var_cases = data["Cases"].var()

n = len(data)

# --------------------------------------------------
# CENTRAL LIMIT THEOREM
# --------------------------------------------------

se = std_cases / np.sqrt(n)

ci_low = mean_cases - 1.96 * se
ci_high = mean_cases + 1.96 * se

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Central Limit Theorem</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Sample means converge to a normal distribution as sample size increases.</div>',
unsafe_allow_html=True
)

st.latex(r"\bar{X} \sim N(\mu,\frac{\sigma}{\sqrt{n}})")

c1,c2,c3 = st.columns(3)

c1.metric("Mean Cases", round(mean_cases,2))
c2.metric("Std Deviation", round(std_cases,2))
c3.metric("Standard Error", round(se,2))

st.write("95% Confidence Interval:", round(ci_low,2), "to", round(ci_high,2))

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# VARIANCE
# --------------------------------------------------

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Variance Analysis</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Variance measures stochastic fluctuation in outbreak intensity.</div>',
unsafe_allow_html=True
)

st.latex(r"Var(X) = \frac{1}{n}\sum (X_i - \mu)^2")

st.metric("Variance", round(var_cases,2))

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# COEFFICIENT OF VARIATION
# --------------------------------------------------

cv = std_cases / mean_cases

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Coefficient of Variation</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Relative variability compared to the mean outbreak level.</div>',
unsafe_allow_html=True
)

st.latex(r"CV = \frac{\sigma}{\mu}")

st.metric("Coefficient of Variation", round(cv,3))

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# GROWTH FACTOR
# --------------------------------------------------

data["growth"] = data["Cases"] / data["Cases"].shift(1)
data["growth"] = data["growth"].replace([np.inf,-np.inf],np.nan)

growth = data["growth"].dropna()

avg_growth = growth.median()

if np.isnan(avg_growth):
    avg_growth = 1.05

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Growth Factor Estimation</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Year-to-year multiplicative growth of dengue cases.</div>',
unsafe_allow_html=True
)

st.latex(r"G_t = \frac{Cases_t}{Cases_{t-1}}")

st.metric("Median Growth Factor", round(avg_growth,3))

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# LYAPUNOV STABILITY
# --------------------------------------------------

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

score = data["dV"].mean()

if score < 0:
    status = "Stable"
elif score < 100000:
    status = "Moderate Risk"
else:
    status = "High Risk"

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Lyapunov Stability</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Determines whether outbreak dynamics stabilize or diverge.</div>',
unsafe_allow_html=True
)

st.latex(r"V(x) = x^2")

st.metric("Stability Status", status)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

simulations = []

last_cases = data["Cases"].iloc[-1]

for i in range(1000):
    
    g = np.random.normal(avg_growth,0.1)
    
    future = last_cases
    
    for j in range(5):
        future = future * g
        
    simulations.append(future)

sim_mean = np.mean(simulations)

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Monte Carlo Simulation</div>', unsafe_allow_html=True)

st.markdown(
'<div class="explain">Simulates future outbreak scenarios using stochastic growth variability.</div>',
unsafe_allow_html=True
)

st.metric("Expected Cases in 5 Years", round(sim_mean))

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FUTURE PREDICTION
# --------------------------------------------------

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

st.header("Future Growth Prediction")

st.latex(r"Cases_{t+1} = Cases_t \times G")

fig = px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True
)

st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# TREND GRAPH
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
# DATA
# --------------------------------------------------

st.header("Dataset")

st.dataframe(data)
