import yfinance as yf
import datetime as dt
import streamlit as st

from streamlit_time_series import *

st.header("Time Series Analysis")

ticker = st.text_input("Please enter ticker here: (for S&P 500 enter ^GSPC)")

frequency_options = ["daily", "weekly", "monthly"]
frequency_box = st.selectbox("select the frequency of prices (default is daily)", frequency_options)

status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))

options = ['historical regime', 'smoothed variance probability', 'continuous wavelet transform', 'all']
series_type = ['Close', 'Adjusted Close']

today = dt.date.today()

before = today - dt.timedelta(days=3653)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

if frequency_box == "daily":
    frequency = "1d"
    
if frequency_box == "weekly":
    frequency = "1wk"
    
if frequency_box == "monthly":
    frequency = "1mo"
    
if status_radio == "Search":

    df = yf.download(ticker, start_date, end_date, interval = frequency)
    df_plot = df[['Close', 'Adj Close']]
    st.line_chart(df_plot)
    st.write(df)

    time_series_type = st.radio("Please select which time series you would like", series_type)
    time_series_options = st.selectbox("Please select what kind of analysis", options)
    time_series_start = st.radio("Please Select Run when ready", ("Stop", "Run"))

    if time_series_start == "Run":
        timeseries = TimeSeries(df, time_series_type, ticker)

        if time_series_options == "historical regime":
            output = timeseries.get_regimes()
            
        if time_series_options == "smoothed variance probability":
            output = timeseries.smoothed_probability()
            
        if time_series_options == "continuous wavelet transform":
            output = timeseries.cwt()
            
        if time_series_options == "all":
            output = timeseries.get_all()

st.write('Disclaimer: Information and output provided on this site does \
         not constitute investment advice.')
st.write('Created by Diego Alvarez')
