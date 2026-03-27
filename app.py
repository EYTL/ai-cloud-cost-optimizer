from lstm_forecast import forecast_lstm
import streamlit as st
import pandas as pd
import plotly.express as px 
from anomaly_model import detect_anomalies

from data_loader import load_data

source = st.sidebar.radio("Data Source", ['nasa', 'aws'])
df = load_data(source)
df = detect_anomalies(df)
st.sidebar.title("Dashboard Controls")
sensitivity = st.sidebar.slider("Anomaly Sensitivity",
                               min_value=1.0,
                               max_value=3.0,
                               value=1.5,
                               step=0.1)

selected_dept= st.sidebar.selectbox("Filter by Department",
                                    ["ALL"] + list(df['department'].unique())) 
if selected_dept!= "ALL":
    df = df[df['department'] ==  selected_dept]
       
st.title("AI Cloud Cost Optimizer")
st.write("Analyzing cloud costs and detecting anomalies in real time.")

st.subheader("Scaling Recommendation")

def scaling_recommendation(values):
    average = sum(values) / len(values)
    if average > 100:
        return 'Scale Up'
    elif average < 20:
        return 'Scale Down'
    else:
        return 'Stable'

if 'count' in df.columns:
    recommendation = scaling_recommendation(df['count'].tolist())
else:
    recommendation = scaling_recommendation(df['cost'].tolist())


if recommendation == 'Scale Up':
    st.error(f"🔴 Recommendation: {recommendation}")
elif recommendation == 'Scale Down':
    st.warning(f"🟡 Recommendation: {recommendation}")
else:
    st.success(f"🟢 Recommendation: {recommendation}")

#---------------------------------------------------------------#

st.subheader("Cost Forecast")

df['forecast'] =  df['cost'].rolling(window=7).mean()

fig3 = px.line(df.tail(1000), x = 'date', y=['cost', 'forecast'],
               title='Actual cost vs 7-Day Moving Average Forecast')   

st.plotly_chart(fig3)

next_predectied = df['forecast'].dropna().iloc[-1]
st.metric("Next Period Predicted Cost", f"${round(next_predectied, 6)}")

def detect_waste(df):
    if 'count' not in df.columns:
        return pd.DataFrame()
    
    
    threshold = df['count'].quantile(0.05)
    waste = df[df['count'] <= threshold]

    return waste

st.subheader("Waste Detector")

if 'count' in df.columns:
    waste = detect_waste(df)
    st.write(f"Wasteful periods detected :{len(waste)}")
    if len(waste) > 0:
        st.warning(f"⚠️{len(waste)} periods show high cost with low traffic")
        st.dataframe(waste.head(20))
else:
    st.info("Waste detection requires traffic count data")

st.subheader("LSTM Neural Network Forecast")

with st.spinner("Training LSTM model on NASA data... (this takes 1-2 minutes)"):
    lstm_prediction = forecast_lstm(df, n_steps=60, epochs=5)

st.metric("LSTM Predicted Next Cost", f"${lstm_prediction:.6f}")
st.caption("Prediction made by a trained LSTM neural network on historical traffic patterns")
    
#-----------------------------------------------------------------------------#

st.subheader("Cost Trend with Anomalies")

fig = px.line(df.tail(1000), x = 'date', y ='cost',
              title='Cost Over the time with Anomalies Highlighed ')
 
anomaly_points = df[df['anomaly'] == -1].tail(1000)

fig.add_scatter(x=anomaly_points['date'],
                y=anomaly_points['cost'],
                mode = 'markers',
                marker = dict(color = 'red' , size = 6),
                name = 'Anomaly')

st.plotly_chart(fig)
        

st.subheader("Executive Summary")

col1, col2, col3 , col4 = st.columns(4)

anomaly_count =  len(df[df['anomaly'] == -1])
waste_count = len(detect_waste(df))

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Anomalies Found", f"{anomaly_count:,}")
col3.metric("Highest Cost", f"${round(df['cost'].max(), 6)}")
col4.metric("Next Period Cost", f"${round(df['forecast'].dropna().iloc[-1], 6)}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Total Spend", f"${df['cost'].sum()}")
col2.metric("Average Daily Cost", f"${round(df['cost'].mean(), 6)}")
col3.metric("Highest Cost", f"${df['cost'].max()}")

st.subheader("Raw Data")
st.dataframe(df)

st.subheader("Anomaly Detection")

anomalies = df[df['anomaly'] == -1 ]
normal = df[df['anomaly'] == 1]

st.write(f"Total records analyzed: {len(df)}")
st.write(f"Anomalies Detected: {len(anomalies)} ({round(len(anomalies)/ len(df)*100, 1)}%)")

if len(anomalies) > 0:
   st.error(f"⚠️ {len(anomalies)} anomalies detected by Isolation Forest!")
   st.dataframe(anomalies.head(20))
else:
    st.success("✅ No anomalies detected")
 
st.download_button(
    label="Download Anomaly Report",
    data = anomalies.to_csv(index=False),
    file_name="anomaly_report.csv",
    mime="text/csv"
)

#-----------------------------------------#



#-----------------------------------------------------------------------------#

st.subheader("Cost by Department")

fig2 =px.bar(df.groupby('department')['cost'].sum().reset_index(),
             x ='department', 
             y = 'cost', 
             color = 'department',
             title='Total Spend per Department')

st.plotly_chart(fig2)


  
     

     
