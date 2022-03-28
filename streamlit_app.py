import streamlit as st
import pandas as pd
import altair as alt
import statsmodels.api as sm

st.set_page_config(
    page_title="time series forecasting", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
## 📊 Time Series Forecasting Dashboard
#### Upload your time series to see the prediction results.
Developed and maintained by [Mohammad Namakshenas](https://www.linkedin.com/in/mohammad-namakshenas-20/)
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
st.markdown("Other indexes can be found in this link: https://www.cryptodatadownload.com/data/binance/. If you "
            "downloaded data from this link, just remove the first row!")
use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

ab_default = None
result_default = None

# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
if use_example_file:
    uploaded_file = "Binance_ETHUSDT_d.csv"
    ab_default = ["variant"]
    result_default = ["converted"]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[::-1].reset_index(drop=True)

    st.markdown("### Data preview")
    st.dataframe(df.head())

    st.markdown("### Data overview")

    df['date'] = pd.to_datetime(df["date"])

    slider_date = st.slider("Enter the visualization period: ",
                            min_value=df['date'].iloc[-1].to_pydatetime(),
                            max_value=df['date'][0].to_pydatetime(),
                            value=[df['date'].iloc[-1].to_pydatetime(), df['date'][0].to_pydatetime()],
                            format="YY-MM-DD")

    df_filtered = df[(df['date'] > slider_date[0]) & (df['date'] < slider_date[1])]

    chart_data = pd.DataFrame(df_filtered['close'])
    chart_data.index = df_filtered['date']

    st.write("")

    st.line_chart(chart_data)

    st.markdown("### Select forecasting method")

    slider_date_pred = st.slider("Enter the in-sample data: ",
                                 min_value=df['date'].iloc[-1].to_pydatetime(),
                                 max_value=df['date'][0].to_pydatetime(),
                                 value=[df['date'].iloc[-1].to_pydatetime(), df['date'][0].to_pydatetime()],
                                 format="YY-MM-DD")

    df_filtered_pred = df[(df['date'] >= slider_date_pred[0]) & (df['date'] <= slider_date_pred[1])]

    select_method = st.selectbox(
        'What is your favorite tool?',
        ('Autoregressive (AR)', 'Moving Average (MA)', 'Mixed Autoregressive Moving Average (ARMA)',
         'Integration (ARIMA)'))
    if select_method == 'Autoregressive (AR)':
        val_p = st.number_input('autoregressive component (p)', min_value=0, max_value=5, value=0, step=1)
        val_q = 0
        val_d = 0
    elif select_method == 'Moving Average (MA)':
        val_q = st.number_input('moving component (q)', min_value=0, max_value=5, value=0, step=1)
        val_p = 0
        val_d = 0
    elif select_method == 'Mixed Autoregressive Moving Average (ARMA)':
        col1, col2 = st.columns(2)
        with col1:
            val_p = st.number_input('autoregressive component (p)', min_value=0, max_value=5, value=0, step=1)
        with col2:
            val_q = st.number_input('moving component (q)', min_value=0, max_value=5, value=0, step=1)
        val_d = 0
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            val_p = st.number_input('autoregressive (p)', min_value=0, max_value=5, value=0, step=1)
        with col2:
            val_q = st.number_input('moving (q)', min_value=0, max_value=5, value=0, step=1)
        with col3:
            val_d = st.number_input('difference (d)', min_value=0, max_value=5, value=0, step=1)

    steps_pred = st.number_input('number of steps to forecast?', min_value=0, max_value=30, value=1, step=1)

    button_pred = st.button("Submit")
    if button_pred:
        model_fit = sm.tsa.arima.ARIMA(df_filtered_pred['close'], order=(val_p, val_d, val_q)).fit()
        pred = model_fit.forecast(steps=steps_pred)
        dates_pred = pd.date_range(start=slider_date_pred[1], periods=steps_pred, freq='D')
        chart_data_pred = pd.DataFrame(pred.to_numpy(), columns=['close'])
        chart_data_pred['date'] = dates_pred

        a = alt.Chart(df_filtered_pred).mark_line(color='blue').encode(
            x='date', y='close')
        b = alt.Chart(chart_data_pred).mark_line(color='red').encode(
            x='date', y='close')
        c = alt.layer(a, b)

        st.markdown("### Forecasting result")

        st.altair_chart(c, use_container_width=True)
        st.markdown("""
        <div style="text-align: center">
        <span style="color:blue">blue line: sampled data</span> / <span style="color:red">red line: predicted data</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        with st.expander("Details (forecasting log)"):
            st.write(model_fit.summary())

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
