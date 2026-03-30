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

# ─────────────────────────────────────────
# SIDEBAR — DATA SOURCE SELECTION
# ─────────────────────────────────────────
st.sidebar.title("Dashboard Controls")

source = st.sidebar.radio(
    "Data Source",
    ['nasa', 'upload', 'aws', 'gcp', 'azure'],
    format_func=lambda x: {
        'nasa': '🛰️ NASA Dataset',
        'upload': '📂 Upload CSV File',
        'aws': '☁️ AWS Cost Explorer',
        'gcp': '🌐 Google Cloud',
        'azure': '🔷 Microsoft Azure'
    }[x]
)

# ─────────────────────────────────────────
# SIDEBAR — CLOUD CREDENTIALS
# ─────────────────────────────────────────
uploaded_file = None
aws_key = aws_secret = aws_region = None
aws_days = 90
gcp_project = gcp_dataset = gcp_table = gcp_sa = None
azure_sub = azure_tenant = azure_client = azure_secret = None

if source == 'upload':
    st.sidebar.markdown("### 📂 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Must have a date column and a cost/amount/count column"
    )
    st.sidebar.caption(
        "Accepted columns: date/time/timestamp + cost/amount/spend/count. "
        "Department auto-detected from service/region columns."
    )

elif source == 'aws':
    st.sidebar.markdown("### ☁️ AWS Credentials")
    with st.sidebar.expander("Enter AWS Credentials", expanded=True):
        aws_key = st.text_input("Access Key ID", type="password",
                                help="Leave blank to use IAM role / env vars")
        aws_secret = st.text_input("Secret Access Key", type="password")
        aws_region = st.selectbox("Region", [
            'us-east-1', 'us-west-2', 'eu-west-1', 'ap-south-1',
            'ap-southeast-1', 'ap-northeast-1', 'ca-central-1'
        ])
        aws_days = st.slider("Days of history", min_value=7, max_value=365, value=90)
    st.sidebar.caption("💡 Cost Explorer must be enabled in your AWS account.")

elif source == 'gcp':
    st.sidebar.markdown("### 🌐 GCP Credentials")
    with st.sidebar.expander("Enter GCP Details", expanded=True):
        gcp_project = st.text_input("Project ID")
        gcp_dataset = st.text_input("BigQuery Dataset")
        gcp_table = st.text_input("Billing Table")
        gcp_sa = st.text_input("Service Account JSON path (optional)")
    st.sidebar.caption("💡 Enable GCP Billing Export to BigQuery first.")

elif source == 'azure':
    st.sidebar.markdown("### 🔷 Azure Credentials")
    with st.sidebar.expander("Enter Azure Details", expanded=True):
        azure_sub = st.text_input("Subscription ID")
        azure_tenant = st.text_input("Tenant ID")
        azure_client = st.text_input("Client ID")
        azure_secret = st.text_input("Client Secret", type="password")
    st.sidebar.caption("💡 Requires Cost Management Reader role.")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = None
load_error = None

try:
    if source == 'upload' and uploaded_file is None:
        st.info("📂 Please upload a CSV file using the sidebar to get started.")
        st.stop()

    df = load_data(
        source=source,
        uploaded_file=uploaded_file,
        aws_access_key=aws_key or None,
        aws_secret_key=aws_secret or None,
        aws_region=aws_region or 'us-east-1',
        aws_days=aws_days,
        gcp_project=gcp_project,
        gcp_dataset=gcp_dataset,
        gcp_table=gcp_table,
        gcp_sa_json=gcp_sa or None,
        azure_sub=azure_sub,
        azure_tenant=azure_tenant,
        azure_client=azure_client,
        azure_secret=azure_secret or None
    )
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"❌ Data load failed: {load_error}")
    st.stop()

df = detect_anomalies(df)

# ─────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────
sensitivity = st.sidebar.slider(
    "Anomaly Sensitivity", min_value=1.0, max_value=3.0, value=1.5, step=0.1
)

selected_dept = st.sidebar.selectbox(
    "Filter by Department / Load",
    ["ALL"] + list(df['department'].unique())
)
if selected_dept != "ALL":
    df = df[df['department'] == selected_dept]

# ─────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────
st.title("🤖 AI Cloud Cost Optimizer")
st.write("Analyzing cloud costs, detecting anomalies, and predicting future spend in real time.")

source_labels = {
    'nasa': 'NASA Traffic Dataset',
    'upload': f'Uploaded File: {uploaded_file.name if uploaded_file else ""}',
    'aws': f'AWS Cost Explorer ({aws_days} days)',
    'gcp': 'Google Cloud Billing',
    'azure': 'Microsoft Azure Cost Management'
}
st.caption(f"📡 Data source: **{source_labels[source]}** | {len(df):,} records loaded")

# ─────────────────────────────────────────
# SECTION 1: AUTOSCALING + LOAD BALANCING
# ─────────────────────────────────────────
st.subheader("⚡ Autoscaling & Load Balancing")

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

# Display scaling recommendation
col_scale1, col_scale2, col_scale3 = st.columns([1, 1, 2])

with col_scale1:
    if recommendation == 'Scale Up':
        st.error(f"🔴 **{recommendation}**\nHigh load detected — provision more resources")
    elif recommendation == 'Scale Down':
        st.warning(f"🟡 **{recommendation}**\nLow utilization — reduce idle resources")
    else:
        st.success(f"🟢 **{recommendation}**\nLoad is balanced — no action needed")

with col_scale2:
    st.metric("Avg Cost (Last 7 Periods)", f"${metrics['avg_cost']:.4f}")
    trend_symbol = "📈" if metrics['cost_trend'] > 0 else "📉"
    st.metric("Cost Trend", f"{trend_symbol} {metrics['cost_trend']:+.4f}")
    if metrics['has_traffic']:
        st.metric("Avg Traffic", f"{metrics['avg_traffic']:.0f} req/min")

with col_scale3:
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
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="% of Total Cost",
            showlegend=False
        )
        st.plotly_chart(fig_lb, use_container_width=True)

# Autoscaling action log
with st.expander("📋 Autoscaling Decision Log"):
    st.markdown(f"""
    | Signal | Value | Status |
    |--------|-------|--------|
    | Avg Cost (7-day) | ${metrics['avg_cost']:.4f} | {'⚠️ High' if metrics['avg_cost'] > 100 else '✅ Normal'} |
    | Cost Trend | {metrics['cost_trend']:+.4f} | {'📈 Rising' if metrics['cost_trend'] > 0 else '📉 Falling'} |
    | Traffic Load | {'N/A' if not metrics['has_traffic'] else f"{metrics['avg_traffic']:.0f}"} | {'N/A' if not metrics['has_traffic'] else ('⚠️ High' if metrics['avg_traffic'] > 100 else '✅ Normal')} |
    | **Final Decision** | **{recommendation}** | {'🔴' if recommendation == 'Scale Up' else '🟡' if recommendation == 'Scale Down' else '🟢'} |
    """)

st.divider()

# ─────────────────────────────────────────
# SECTION 2: COST FORECAST (Moving Average)
# ─────────────────────────────────────────
st.subheader("📈 Cost Forecast")

window = min(7, len(df) // 2) if len(df) < 14 else 7
df['forecast'] = df['cost'].rolling(window=window).mean()

fig3 = px.line(
    df.tail(1000), x='date', y=['cost', 'forecast'],
    title=f'Actual Cost vs {window}-Period Moving Average Forecast'
)
st.plotly_chart(fig3, use_container_width=True)

next_predicted = df['forecast'].dropna().iloc[-1]
st.metric("Next Period Predicted Cost (Moving Avg)", f"${round(next_predicted, 6)}")

st.divider()

# ─────────────────────────────────────────
# SECTION 3: LSTM FORECAST
# ─────────────────────────────────────────
st.subheader("🧠 LSTM Neural Network Forecast")

data_size = len(df)
if data_size < 5:
    st.warning("⚠️ Not enough data for LSTM (minimum 5 records). Add more data or use NASA dataset.")
else:
    with st.spinner(f"Training LSTM on {data_size} records... (may take 1-2 mins for large datasets)"):
        try:
            lstm_prediction = forecast_lstm(df, n_steps=60, epochs=5)
            st.metric("LSTM Predicted Next Cost", f"${lstm_prediction:.6f}")
            st.caption(
                f"Trained on {data_size} records. "
                + ("Note: LSTM auto-adjusted sequence length due to limited data rows." if data_size < 60 else "")
            )
        except Exception as e:
            st.error(f"LSTM failed: {e}")

st.divider()

# ─────────────────────────────────────────
# SECTION 4: WASTE DETECTOR
# ─────────────────────────────────────────
st.subheader("🗑️ Waste Detector")

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
st.write(f"Wasteful periods detected: **{len(waste)}**")
if len(waste) > 0:
    st.warning(f"⚠️ {len(waste)} periods show unusually low utilization / spend")
    st.dataframe(waste.head(20))
else:
    st.success("✅ No waste detected")

st.divider()

# ─────────────────────────────────────────
# SECTION 5: ANOMALY DETECTION
# ─────────────────────────────────────────
st.subheader("🔍 Cost Trend with Anomalies")

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
    st.error(f"⚠️ {len(anomalies)} anomalies detected by Isolation Forest!")
    st.dataframe(anomalies.head(20))
else:
    st.success("✅ No anomalies detected")

st.download_button(
    label="⬇️ Download Anomaly Report",
    data=anomalies.to_csv(index=False),
    file_name="anomaly_report.csv",
    mime="text/csv"
)

st.divider()

# ─────────────────────────────────────────
# SECTION 6: EXECUTIVE SUMMARY
# ─────────────────────────────────────────
st.subheader("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)
anomaly_count = len(anomalies)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Anomalies Found", f"{anomaly_count:,}")
col3.metric("Highest Cost", f"${round(df['cost'].max(), 6)}")
col4.metric("Next Period Cost (MA)", f"${round(df['forecast'].dropna().iloc[-1], 6)}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Total Spend", f"${round(df['cost'].sum(), 4)}")
col2.metric("Average Daily Cost", f"${round(df['cost'].mean(), 6)}")
col3.metric("Waste Periods", f"{len(waste):,}")

st.divider()

# ─────────────────────────────────────────
# SECTION 7: COST BY DEPARTMENT
# ─────────────────────────────────────────
st.subheader("🏢 Cost by Department / Service")

fig2 = px.bar(
    df.groupby('department')['cost'].sum().reset_index(),
    x='department', y='cost', color='department',
    title='Total Spend per Department / Service'
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# SECTION 8: RAW DATA
# ─────────────────────────────────────────
st.subheader("📋 Raw Data")
st.dataframe(df)