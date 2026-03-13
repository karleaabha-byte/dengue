import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Dengue Outbreak Dynamics",
    page_icon="🦟",
    layout="wide"
)

# --------------------------------------------------
# CSS (APP STYLE LOOK)
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:5px;
}

.subtitle {
text-align:center;
color:gray;
margin-bottom:30px;
}

div[data-testid="stMetric"]{
background-color:white;
border-radius:15px;
padding:10px;
box-shadow:0px 4px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="main-title">Dengue Outbreak Dynamics</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Statistical and Stochastic Analysis of Dengue Cases</div>',
unsafe_allow_html=True
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

st.sidebar.header("Controls")

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
"Select Region",
regions
)

data = df[df["Region"] == region].sort_values("Year")

# --------------------------------------------------
# BASIC CALCULATIONS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
variance = data["Cases"].var()

data["growth"] = data["Cases"].pct_change()
growth = data["growth"].replace([np.inf,-np.inf],np.nan).dropna()

avg_growth = growth.median() if len(growth)>0 else 0

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

lyapunov = data["dV"].mean()

data["rolling_avg"] = data["Cases"].rolling(3).mean()

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Mean Cases", round(mean_cases,1))
col2.metric("Std Dev", round(std_cases,1))
col3.metric("Variance", round(variance,1))
col4.metric("Growth Rate", round(avg_growth,3))

# --------------------------------------------------
# CENTRAL LIMIT THEOREM
# --------------------------------------------------

st.header("Central Limit Theorem")

st.write(
"As sample size increases, the sampling distribution of the mean approaches a normal distribution."
)

st.latex(r"\bar{X} \sim N(\mu,\frac{\sigma}{\sqrt{n}})")

# --------------------------------------------------
# VARIANCE
# --------------------------------------------------

st.header("Variance")

st.write("Variance measures dispersion in dengue case counts.")

st.latex(r"Var(X)=\frac{1}{n}\sum (X_i-\mu)^2")

st.write("Variance:", round(variance,2))

# --------------------------------------------------
# COEFFICIENT OF VARIATION
# --------------------------------------------------

st.header("Coefficient of Variation")

st.write("Relative variability compared to the mean.")

st.latex(r"CV=\frac{\sigma}{\mu}")

cv = std_cases/mean_cases if mean_cases!=0 else np.nan

st.write("CV:", round(cv,3))

# --------------------------------------------------
# LYAPUNOV STABILITY
# --------------------------------------------------

st.header("Lyapunov Stability")

st.write("Used to evaluate stability of outbreak dynamics.")

st.latex(r"V(x)=x^2")

c1,c2 = st.columns(2)

c1.metric("Lyapunov Value", round(lyapunov,2))

if lyapunov < 0:
    status = "Stable"
elif lyapunov < 100000:
    status = "Moderate Risk"
else:
    status = "Unstable"

c2.metric("System Stability", status)

# --------------------------------------------------
# YEARWISE BAR CHART
# --------------------------------------------------

st.header("Year-wise Dengue Cases")

fig_bar = px.bar(
data,
x="Year",
y="Cases",
color="Cases",
color_continuous_scale="RdPu"
)

fig_bar.update_layout(
template="plotly_white"
)

st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# AREA OUTBREAK GRAPH
# --------------------------------------------------

st.header("Outbreak Intensity Over Time")

fig_area = px.area(
data,
x="Year",
y="Cases",
color_discrete_sequence=["#ff4da6"]
)

fig_area.update_layout(template="plotly_white")

st.plotly_chart(fig_area, use_container_width=True)

# --------------------------------------------------
# ROLLING TREND
# --------------------------------------------------

st.header("Smoothed Trend (3-Year Moving Average)")

fig_trend = go.Figure()

fig_trend.add_trace(
go.Scatter(
x=data["Year"],
y=data["Cases"],
mode="lines+markers",
name="Actual Cases",
line=dict(color="#ff4da6")
)
)

fig_trend.add_trace(
go.Scatter(
x=data["Year"],
y=data["rolling_avg"],
mode="lines",
name="3-Year Avg",
line=dict(color="#7a0177", width=4)
)
)

fig_trend.update_layout(template="plotly_white")

st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

st.header("Monte Carlo Outbreak Simulation")

st.write("Simulates possible future outbreak trajectories.")

st.latex(r"Cases_{t+1}=Cases_t(1+G+\epsilon)")

last_cases = data["Cases"].iloc[-1]
last_year = int(data["Year"].max())

future_years = 5
simulations = 200

years = list(range(last_year+1,last_year+future_years+1))

paths = []

for s in range(simulations):

    current = last_cases
    path=[]

    for y in years:

        noise = np.random.normal(0,0.05)

        current = current*(1+avg_growth+noise)

        path.append(current)

    paths.append(path)

paths=np.array(paths)

mean_path=paths.mean(axis=0)
upper=np.percentile(paths,95,axis=0)
lower=np.percentile(paths,5,axis=0)

fig_sim=go.Figure()

fig_sim.add_trace(go.Scatter(x=years,y=upper,line=dict(width=0),showlegend=False))

fig_sim.add_trace(
go.Scatter(
x=years,
y=lower,
fill="tonexty",
fillcolor="rgba(255,77,166,0.2)",
line=dict(width=0),
name="Uncertainty Range"
)
)

fig_sim.add_trace(
go.Scatter(
x=years,
y=mean_path,
mode="lines+markers",
line=dict(color="#ff4da6",width=4),
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

st.latex(r"Cases_{t+1}=Cases_t(1+G)")

future=[]
current=last_cases

for i in range(1,6):

    current=current*(1+avg_growth)

    future.append({
    "Year": last_year+i,
    "Cases": current
    })

future_df=pd.DataFrame(future)

combined=pd.concat([data[["Year","Cases"]],future_df])

combined["Type"]=["Actual"]*len(data)+["Predicted"]*len(future_df)

fig_pred=px.line(
combined,
x="Year",
y="Cases",
color="Type",
markers=True,
color_discrete_sequence=["#ff4da6","#ff99cc"]
)

fig_pred.update_layout(template="plotly_white")

st.plotly_chart(fig_pred,use_container_width=True)

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.header("Dataset")

st.dataframe(data)
