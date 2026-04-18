from lstm_forecast import forecast_lstm
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anomaly_model import detect_anomalies
from data_loader import load_data

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="AI Cloud Cost Optimizer", layout="wide")

st.markdown("""
<style>
    /* Gradient Title */
    h1 {
        padding-bottom: 2rem !important;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Section Header Partitions */
    h3 {
        margin-top: 3.5rem !important;
        margin-bottom: 2rem !important;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        color: #0ea5e9;
        font-weight: 600 !important;
    }
    
    /* Subtle Dividers */
    hr {
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        border-color: rgba(128, 128, 128, 0.2) !important;
    }
    
    /* Floating Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Make metric values pop */
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #0ea5e9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR — DATA SOURCE SELECTION
# ─────────────────────────────────────────
st.sidebar.title("Dashboard Controls")

source_container = st.sidebar.container()
filter_container = st.sidebar.container()
creds_container = st.sidebar.container()


source = source_container.radio(
    "Data Source",
    ['nasa', 'upload', 'aws'],
    format_func=lambda x: {
        'nasa': 'NASA Dataset',
        'upload': 'Upload CSV File',
        'aws': 'AWS Cost Explorer'
    }[x]
)

# ─────────────────────────────────────────
# SIDEBAR — CLOUD CREDENTIALS
# ─────────────────────────────────────────
uploaded_file = None
aws_key = aws_secret = aws_region = None
aws_days = 90

if source == 'upload':
    creds_container.markdown("### Upload Your Data")
    uploaded_file = creds_container.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Needs a date column and either a spend/cost column or a usage/count column"
    )
    creds_container.caption(
        "Accepted date columns include date/time/timestamp. "
        "Spend columns like Total Cost, cost_usd, Unblended Cost, amount, spend, charge, price, fee, and count-like usage columns are auto-detected. "
        "Department is auto-detected from service/region columns."
    )

elif source == 'aws':
    creds_container.markdown("### AWS Credentials")
    with creds_container.expander("Enter AWS Credentials", expanded=True):
        aws_key = st.text_input("Access Key ID", type="password",
                                help="Leave blank to use IAM role / env vars")
        aws_secret = st.text_input("Secret Access Key", type="password")
        aws_region = st.selectbox("Region", [
            'us-east-1', 'us-west-2', 'eu-west-1', 'ap-south-1',
            'ap-southeast-1', 'ap-northeast-1', 'ca-central-1'
        ])
        aws_days = st.slider("Days of history", min_value=7, max_value=365, value=90)
    creds_container.caption("Cost Explorer must be enabled in your AWS account.")



# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = None
load_error = None

try:
    if source == 'upload' and uploaded_file is None:
        st.info("Please upload a CSV file using the sidebar to get started.")
        st.stop()

    df = load_data(
        source=source,
        uploaded_file=uploaded_file,
        aws_access_key=aws_key or None,
        aws_secret_key=aws_secret or None,
        aws_region=aws_region or 'us-east-1',
        aws_days=aws_days
    )
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"Data load failed: {load_error}")
    st.stop()

df = detect_anomalies(df)

# ─────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────
sensitivity = filter_container.slider(
    "Anomaly Sensitivity", min_value=1.0, max_value=3.0, value=1.5, step=0.1
)

selected_dept = filter_container.selectbox(
    "Filter by Department / Load",
    ["ALL"] + list(df['department'].unique())
)

# Invisible spacer to ensure dropdowns always open downwards
st.sidebar.markdown("<div style='height: 60vh; visibility: hidden;'>Spacer</div>", unsafe_allow_html=True)


if selected_dept != "ALL":
    df = df[df['department'] == selected_dept]

# ─────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────
st.title("AI Cloud Cost Optimizer")
st.write("Analyzing cloud costs, detecting anomalies, and predicting future spend in real time.")

source_labels = {
    'nasa': 'NASA Traffic Dataset',
    'upload': f'Uploaded File: {uploaded_file.name if uploaded_file else ""}',
    'aws': f'AWS Cost Explorer ({aws_days} days)'
}
st.caption(f"Data source: **{source_labels[source]}** | {len(df):,} records loaded")

# ─────────────────────────────────────────
# SECTION 1: AUTOSCALING + LOAD BALANCING
# ─────────────────────────────────────────
st.markdown("### Autoscaling & Load Balancing")

def get_scaling_metrics(df):
    """Compute scaling signals from cost and traffic data."""
    recent = df.tail(7)  # last 7 periods
    avg_cost = recent['cost'].mean()
    max_cost = recent['cost'].max()
    cost_trend = df['cost'].tail(14).mean() - df['cost'].tail(30).mean()

    # Traffic signal (NASA has count, AWS doesn't)
    has_traffic = 'count' in df.columns
    avg_traffic = recent['count'].mean() if has_traffic else None
    traffic_trend = (
        df['count'].tail(7).mean() - df['count'].tail(14).mean()
        if has_traffic else None
    )

    return {
        'avg_cost': avg_cost,
        'max_cost': max_cost,
        'cost_trend': cost_trend,
        'has_traffic': has_traffic,
        'avg_traffic': avg_traffic,
        'traffic_trend': traffic_trend
    }

def scaling_recommendation(metrics):
    """Rule-based + trend-aware scaling recommendation."""
    signals = []

    # Cost-based signals
    if metrics['avg_cost'] > 100 or metrics['cost_trend'] > 10:
        signals.append('scale_up')
    elif metrics['avg_cost'] < 20 and metrics['cost_trend'] < 0:
        signals.append('scale_down')
    else:
        signals.append('stable')

    # Traffic-based signals (if available)
    if metrics['has_traffic']:
        if metrics['avg_traffic'] > 100 or metrics['traffic_trend'] > 20:
            signals.append('scale_up')
        elif metrics['avg_traffic'] < 20 and metrics['traffic_trend'] < 0:
            signals.append('scale_down')

    # Majority vote
    if signals.count('scale_up') > signals.count('scale_down'):
        return 'Scale Up'
    elif signals.count('scale_down') > signals.count('scale_up'):
        return 'Scale Down'
    else:
        return 'Stable'

def load_balancer_status(df):
    """Simulate load balancer distribution across departments."""
    dept_costs = df.groupby('department')['cost'].sum()
    total = dept_costs.sum()
    if total == 0:
        return {}
    distribution = (dept_costs / total * 100).round(1).to_dict()
    return distribution

metrics = get_scaling_metrics(df)
recommendation = scaling_recommendation(metrics)
lb_distribution = load_balancer_status(df)

# Display Metrics and Recommendation side-by-side
m1, m2, m3, m4 = st.columns([1, 1, 1, 2])

with m1:
    st.metric("Avg Cost (7 Days)", f"${metrics['avg_cost']:.4f}")

with m2:
    trend_symbol = "↑" if metrics['cost_trend'] > 0 else "↓"
    st.metric("Cost Trend", f"{trend_symbol} {metrics['cost_trend']:+.4f}")

with m3:
    if metrics['has_traffic']:
        st.metric("Avg Traffic", f"{metrics['avg_traffic']:.0f} req/m")
    else:
        st.metric("Avg Traffic", "N/A")

with m4:
    if recommendation == 'Scale Up':
        st.error(f"**{recommendation}**\nHigh load detected — provision more resources")
    elif recommendation == 'Scale Down':
        st.warning(f"**{recommendation}**\nLow utilization — reduce idle resources")
    else:
        st.success(f"**{recommendation}**\nLoad is balanced — no action needed")

st.write("")

# Display Load Balancer full width
if lb_distribution:
    st.markdown("**Load Balancer Distribution**")
    fig_lb = go.Figure(go.Bar(
        x=list(lb_distribution.values()),
        y=list(lb_distribution.keys()),
        orientation='h',
        marker_color=[
            '#ef4444' if v > 50 else '#f59e0b' if v > 25 else '#22c55e'
            for v in lb_distribution.values()
        ]
    ))
    fig_lb.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="% of Total Cost",
        showlegend=False
    )
    st.plotly_chart(fig_lb, use_container_width=True)

# Autoscaling action log
with st.expander("Autoscaling Decision Log"):
    st.markdown(f"""
    | Signal | Value | Status |
    |--------|-------|--------|
    | Avg Cost (7-day) | ${metrics['avg_cost']:.4f} | {'High' if metrics['avg_cost'] > 100 else 'Normal'} |
    | Cost Trend | {metrics['cost_trend']:+.4f} | {'Rising' if metrics['cost_trend'] > 0 else 'Falling'} |
    | Traffic Load | {'N/A' if not metrics['has_traffic'] else f"{metrics['avg_traffic']:.0f}"} | {'N/A' if not metrics['has_traffic'] else ('High' if metrics['avg_traffic'] > 100 else 'Normal')} |
    | **Final Decision** | **{recommendation}** | {'🔴' if recommendation == 'Scale Up' else '🟡' if recommendation == 'Scale Down' else '🟢'} |
    """)

st.write("")

# ─────────────────────────────────────────
# SECTION 2: COST FORECAST (Moving Average)
# ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Cost Forecast")

    window = min(7, len(df) // 2) if len(df) < 14 else 7
    df['forecast'] = df['cost'].rolling(window=window).mean()

    next_predicted = df['forecast'].dropna().iloc[-1]

    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig3 = px.line(
            df.tail(1000), x='date', y=['cost', 'forecast'],
            title=f'Actual Cost vs {window}-Period Moving Average Forecast'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.metric("Next Period Predicted Cost (Moving Avg)", f"${next_predicted:.6f}")
        
    st.write("")

# ─────────────────────────────────────────
# SECTION 3: LSTM FORECAST
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_cached_lstm_forecast(data, n_steps, epochs):
    return forecast_lstm(data, n_steps=n_steps, epochs=epochs)

with st.container(border=True):
    st.markdown("### LSTM Neural Network Forecast")

    data_size = len(df)
    if data_size < 5:
        st.warning("Not enough data for LSTM (minimum 5 records). Add more data or use NASA dataset.")
    else:
        with st.spinner(f"Training LSTM on {data_size} records... (may take 1-2 mins for large datasets)"):
            try:
                lstm_prediction = get_cached_lstm_forecast(df, n_steps=60, epochs=5)
                st.metric("LSTM Predicted Next Cost", f"${lstm_prediction:.6f}")
                st.caption(
                    f"Trained on {data_size} records. "
                    + ("Note: LSTM auto-adjusted sequence length due to limited data rows." if data_size < 60 else "")
                )
            except Exception as e:
                st.error(f"LSTM failed: {e}")

    st.write("")

# ─────────────────────────────────────────
# SECTION 4: WASTE DETECTOR
# ─────────────────────────────────────────
def detect_waste(df):
    if 'count' not in df.columns:
        # For cloud data without traffic: flag low-cost high-variance periods
        threshold = df['cost'].quantile(0.05)
        waste = df[df['cost'] <= threshold]
        return waste
    threshold = df['count'].quantile(0.05)
    waste = df[df['count'] <= threshold]
    return waste

waste = detect_waste(df)

with st.container(border=True):
    st.markdown("### Waste Detector")
    st.write(f"Wasteful periods detected: **{len(waste)}**")
    if len(waste) > 0:
        st.warning(f"{len(waste)} periods show unusually low utilization / spend")
        st.dataframe(waste.head(20))
    else:
        st.success("No waste detected")

    st.write("")

# ─────────────────────────────────────────
# SECTION 5: ANOMALY DETECTION
# ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Cost Trend with Anomalies")

    fig = px.line(
        df.tail(1000), x='date', y='cost',
        title='Cost Over Time with Anomalies Highlighted'
    )
    anomaly_points = df[df['anomaly'] == -1].tail(1000)
    fig.add_scatter(
        x=anomaly_points['date'],
        y=anomaly_points['cost'],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomaly'
    )
    st.plotly_chart(fig, use_container_width=True)

    anomalies = df[df['anomaly'] == -1]
    normal = df[df['anomaly'] == 1]

    st.write(f"Total records analyzed: **{len(df)}**")
    st.write(f"Anomalies Detected: **{len(anomalies)}** ({round(len(anomalies) / len(df) * 100, 1)}%)")

    if len(anomalies) > 0:
        st.error(f"{len(anomalies)} anomalies detected by Isolation Forest!")
        st.dataframe(anomalies.head(20))
    else:
        st.success("No anomalies detected")

    st.download_button(
        label="Download Anomaly Report",
        data=anomalies.to_csv(index=False),
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

    st.write("")

# ─────────────────────────────────────────
# SECTION 6: EXECUTIVE SUMMARY
# ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${df['cost'].sum():,.2f}")
    col2.metric("Average Daily Cost", f"${df['cost'].mean():,.4f}")
    col3.metric("Highest Cost", f"${df['cost'].max():,.4f}")
    col4.metric("Next Period (MA)", f"${df['forecast'].dropna().iloc[-1]:,.4f}")

    st.write("")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Records Analyzed", f"{len(df):,}")
    
    anomaly_count = len(anomalies)
    col6.metric("Anomalies Found", f"{anomaly_count:,}", 
                delta=f"{anomaly_count} detected" if anomaly_count else None, 
                delta_color="inverse")
    
    waste_count = len(waste)
    col7.metric("Wasteful Periods", f"{waste_count:,}", 
                delta=f"{waste_count} inactive" if waste_count else None, 
                delta_color="inverse")
                
    health_status = "Optimal" if anomaly_count <= 2 and waste_count <= 5 else "Needs Review"
    if anomaly_count == 0 and waste_count == 0:
        health_status = "Perfect"
    col8.metric("System Health", health_status)

    st.write("")

# ─────────────────────────────────────────
# SECTION 7: COST BY DEPARTMENT
# ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Cost by Department / Service")

    fig2 = px.bar(
        df.groupby('department')['cost'].sum().reset_index(),
        x='department', y='cost', color='department',
        title='Total Spend per Department / Service'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("")

# ─────────────────────────────────────────
# SECTION 8: RAW DATA
# ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Raw Data")
    st.dataframe(df)
