import yfinance as yf
import datetime as dt
import streamlit as st

from streamlit_markov_regime import *
from streamlit_time_series import *

st.header("Time Series Analysis")

sidebar_options = ['tools', 'expirement']
frequency_options = ["daily", "weekly", "monthly",]
sidebar = st.sidebar.selectbox("Select Option", sidebar_options)

today = dt.date.today()

before = today - dt.timedelta(days=3653)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')
    
if sidebar == "tools":
    
    search_col1, search_col2 = st.beta_columns(2)
    
    with search_col1:
        ticker = st.text_input("Please enter ticker here:")
        
    with search_col2:
        frequency = st.selectbox("Select Frequency", frequency_options)
        
        if frequency == "daily":
            interval = "1d"
        
        if frequency == "weekly":
            interval = "1wk"
            
        if frequency == "monthly":
            interval = "1mo"
    
    status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))
    
    options = ['historical regime', 'smoothed variance probability', 'continuous wavelet transform', 'all']
    series_type = ['Close', 'Adjusted Close']

    if status_radio == "Search":
    
        df = yf.download(ticker, start_date, end_date, interval = interval)
        df_plot = df[['Close', 'Adj Close']]
        st.line_chart(df_plot)
        st.write(df)
    
        time_series_type = st.radio("Please select which time series you would like", series_type)
        time_series_options = st.selectbox("Please select what kind of analysis", options)
        time_series_start = st.radio("Please Select Run when ready", ("Stop", "Run"))
    
        if time_series_start == "Run":
            timeseries = TimeSeries(df, time_series_type, ticker, frequency)
    
            if time_series_options == "historical regime":
                output = timeseries.get_regimes()
                
            if time_series_options == "smoothed variance probability":
                output = timeseries.smoothed_probability()
                
            if time_series_options == "continuous wavelet transform":
                output = timeseries.cwt()
                
            if time_series_options == "all":
                output = timeseries.get_all()

if sidebar == "expirement":
    
    frequency_list = ['daily', 'weekly']
    price_type_list = ['Close', 'Adj Close']
    
    ticker = st.text_input("Please enter ticker here:")
    option_col1, option_col2, option_col3 = st.beta_columns(3)
    
    with option_col1:
        status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))
        
    with option_col2:
        frequency = st.selectbox("choose frequency", frequency_list)
        
    with option_col3:
        price_type = st.selectbox("choose price type", price_type_list)
    
    options = ['historical regime', 'smoothed variance probability', 'continuous wavelet transform', 'all']
    series_type = ['Close', 'Adjusted Close']

    if status_radio == "Search":
        
        df = Prices(start_date, end_date, ticker, price_type, frequency).getPrices()
        st.dataframe(df)
        st.subheader("{} ".format(ticker) + "{} ".format(price_type) + "{} ".format(frequency) + "price")
        st.line_chart(df)
        
        expirement_options = ["indicator function / volatility filtering"]
        expirement_method = st.selectbox("select expirement", expirement_options)
        
        if expirement_method == "indicator function / volatility filtering":
            
            indicator_run = st.radio("select run once ready", ("Stop", "Run"))
            
            if indicator_run == "Run":
                
                output_df = Markov(df, ticker, frequency, price_type).get_indicator()
                st.line_chart(output_df)
        
        

st.write('Disclaimer: Information and output provided on this site does \
         not constitute investment advice.')
st.write('Created by Diego Alvarez')
