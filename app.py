import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Dengue Dynamics",
    page_icon="🦟",
    layout="wide"
)

# --------------------------------------------------
# COOL CSS (THIS MAKES IT LOOK LIKE AN APP)
# --------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #f6f7fb;
}

.main-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    margin-bottom:10px;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:40px;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

.section-card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
'<div class="main-title">Dengue Outbreak Dynamics</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Statistical + Stochastic Analysis of Dengue Cases</div>',
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

st.sidebar.title("Controls")

regions = sorted(df["Region"].unique())

region = st.sidebar.selectbox(
"Select Region",
regions
)

data = df[df["Region"] == region].sort_values("Year")

# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------

mean_cases = data["Cases"].mean()
std_cases = data["Cases"].std()
variance = data["Cases"].var()

data["growth"] = data["Cases"].pct_change()
growth = data["growth"].dropna()

avg_growth = growth.median() if len(growth)>0 else 0

data["V"] = data["Cases"]**2
data["dV"] = data["V"].diff()

lyapunov = data["dV"].mean()

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Mean Cases", round(mean_cases,1))
col2.metric("Std Dev", round(std_cases,1))
col3.metric("Variance", round(variance,1))
col4.metric("Growth Rate", round(avg_growth,3))

# --------------------------------------------------
# TREND
# --------------------------------------------------

st.markdown("### Case Trend")

fig = px.line(
data,
x="Year",
y="Cases",
markers=True,
color_discrete_sequence=["#ff4da6"]
)

fig.update_layout(
template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# LYAPUNOV
# --------------------------------------------------

st.markdown("### System Stability")

c1,c2 = st.columns(2)

c1.metric("Lyapunov Value", round(lyapunov,2))

if lyapunov < 0:
    status="Stable"
elif lyapunov<100000:
    status="Moderate"
else:
    status="Unstable"

c2.metric("Outbreak Stability", status)

# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

st.markdown("### Future Simulation")

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

fig_sim.add_trace(
go.Scatter(
x=years,
y=upper,
line=dict(width=0),
showlegend=False
))

fig_sim.add_trace(
go.Scatter(
x=years,
y=lower,
fill="tonexty",
fillcolor="rgba(255,77,166,0.2)",
line=dict(width=0),
name="Uncertainty"
))

fig_sim.add_trace(
go.Scatter(
x=years,
y=mean_path,
line=dict(color="#ff4da6",width=4),
mode="lines+markers",
name="Expected"
))

fig_sim.update_layout(
template="plotly_white",
xaxis_title="Year",
yaxis_title="Predicted Cases"
)

st.plotly_chart(fig_sim,use_container_width=True)

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------

st.markdown("### Case Distribution")

fig_hist = px.histogram(
data,
x="Cases",
nbins=20,
color_discrete_sequence=["#ff80bf"]
)

fig_hist.update_layout(template="plotly_white")

st.plotly_chart(fig_hist,use_container_width=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------

st.markdown("### Dataset")

st.dataframe(data)
