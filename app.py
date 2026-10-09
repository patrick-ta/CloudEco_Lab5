import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Amazon EC2 Cost App", layout="wide")

COST_COLS = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost',
             'Windows On Demand cost', 'Windows Reserved cost']
STATS = ['mean', '25%', '50%', '75%']
FEATURES = ['Instance Memory', 'vCPUs']

@st.cache_data
def load():
    data = pd.read_csv('ec2dataset.csv')
    for col in COST_COLS:
        data[col] = pd.to_numeric(
            data[col].str.replace('[$, hourly]', '', regex=True), errors='coerce')
    data['Instance Memory'] = pd.to_numeric(
        data['Instance Memory'].str.replace(' GiB', ''), errors='coerce')
    data['vCPUs'] = pd.to_numeric(
        data['vCPUs'].str.extract(r'(\d+)', expand=False), errors='coerce')
    return data


@st.cache_resource
def train(data):
    clean = data.dropna(subset=['On Demand'] + FEATURES)
    X, y = clean[FEATURES], clean['On Demand']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return model, y_test, model.predict(X_test)


def stats_chart(df):
    summary = df[COST_COLS].describe()
    st.bar_chart(summary.loc[STATS].T)
    st.dataframe(summary)

def analysis_tab(data):
    st.subheader("Cost comparison (all instances)")
    stats_chart(data)

    st.subheader("On-Demand outliers (IQR method)")
    q1, q3 = data['On Demand'].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = data[(data['On Demand'] < q1 - 1.5 * iqr) |
                    (data['On Demand'] > q3 + 1.5 * iqr)]
    st.dataframe(outliers)

    st.subheader("Most cost-effective instances")
    n = st.slider("Number of instances", 5, 50, 10)
    cheapest = (data[['Name', 'On Demand', 'Linux Reserved cost']]
                .dropna().sort_values('On Demand').head(n).set_index('Name'))
    st.bar_chart(cheapest)

    st.subheader("Instance family comparison")
    for fam in st.multiselect("Families", ['T2', 'T3'], default=['T2', 'T3']):
        st.markdown(f"**{fam} instances**")
        stats_chart(data[data['Name'].str.startswith(fam)])


def predictor_tab(data):
    model, y_test, y_pred = train(data)

    st.subheader("Model performance")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")

    st.scatter_chart(pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}),
                     x='Actual', y='Predicted')

    st.subheader("Predict cost")
    mem = st.number_input("Memory (GiB)", min_value=0.0, value=4.0)
    cpu = st.number_input("vCPUs", min_value=1, value=2)
    pred = model.predict(pd.DataFrame([[mem, cpu]], columns=FEATURES))[0]
    st.success(f"Predicted On-Demand Cost: ${pred:.4f}/hr")

data = load()
st.title("Amazon EC2 Cost Analysis & Predictor")
st.subheader("Student: Patrick Ta")
st.subheader("ID: 991677975")

tab1, tab2 = st.tabs(["📊 Cost Analysis", "🤖 Cost Predictor"])
with tab1:
    analysis_tab(data)
with tab2:
    predictor_tab(data)