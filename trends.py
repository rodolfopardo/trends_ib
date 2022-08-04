import streamlit as st
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(page_title="Infobae Trends", page_icon="📈",
                   layout="wide")  # needs to be the first thing after the streamlit import
st.set_option('deprecation.showPyplotGlobalUse', False)
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from pytrends.request import TrendReq

set_random_seed(0)

st.write(
    "Realizado por el equipo de Producto de Infobae")
st.title("Predicción de una keyword en Google Trends")

# streamlit variables
KW = st.text_input('Escriba la keyword')
KW = [KW]
FORECAST_WEEKS = st.sidebar.text_input('Número de semanas a predecir', value=10)
LANGUAGE = st.sidebar.selectbox(
    "Elija el país según las siglas de Google",
    (
        "AR",
        "US",
        "PE",
        "CO",
        "MX",
    ),
)
RETRIES = st.sidebar.text_input('Dejar valor 3', value=3)
HISTORIC = st.sidebar.checkbox('Marque para correr prediccion', value=True)
RETRIES = int(RETRIES)
FORECAST_WEEKS = int(FORECAST_WEEKS)

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Correr prediccion')

if submitted:
    st.write("Buscando y generando predicción de: %s" % KW[0])
    pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5)

    pt.build_payload(KW, timeframe='today 12-m', geo=LANGUAGE)
    df = pt.interest_over_time()

    df = df[df['isPartial'] == False].reset_index()
    data = df.rename(columns={'date': 'ds', KW[0]: 'y'})[['ds', 'y']]
    model = NeuralProphet(daily_seasonality=True)
    metrics = model.fit(data, freq="W")

    future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)

    data = model.predict(future)
    data = data.rename(columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'})[['date', 'actual', 'predicted']]

    forecast = model.predict(future)
    ax = model.plot(forecast, ylabel='Búsquedas', xlabel='Año', figsize=(14, 9))
    st.subheader(KW[0])

    @st.cache
    def convert_df(df):  # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(data)

    st.download_button(
        label="📥 Baje las predicciones!",
        data=csv,
        file_name='gtrends_predicciones.csv',
        mime='text/csv', )

    st.pyplot()
    st.write(data)