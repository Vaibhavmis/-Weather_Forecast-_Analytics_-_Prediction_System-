import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# Streamlit Configuration


st.set_page_config(
    page_title="Weather Forecast Analytics & Prediction System",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Forecast Analytics & Prediction System")
st.markdown("---")


# Load Dataset


@st.cache_data
def load_data():
    df = pd.read_csv("weatherHistory.csv")
    return df

df = load_data()

# Data Preprocessing


df.columns = df.columns.str.strip()

# Convert Date Column

df["Formatted Date"] = pd.to_datetime(
    df["Formatted Date"],
    utc=True
)

# Remove Missing Values

df = df.dropna()

# Remove Duplicate Rows

df = df.drop_duplicates()

# Rename Columns for Easy Access

df.rename(columns={
    "Temperature (C)": "Temperature",
    "Apparent Temperature (C)": "Apparent_Temperature",
    "Humidity": "Humidity",
    "Wind Speed (km/h)": "WindSpeed",
    "Wind Bearing (degrees)": "WindBearing",
    "Visibility (km)": "Visibility",
    "Pressure (millibars)": "Pressure"
}, inplace=True)


# Sidebar


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Temperature Analytics",
        "Humidity Analytics",
        "Wind Analytics",
        "Pressure Analytics",
        "Weather Trends",
        "Machine Learning"
    ]
)


# Dashboard


if page == "Dashboard":

    st.header("Weather Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        len(df)
    )

    c2.metric(
        "Average Temperature",
        f"{df['Temperature'].mean():.2f} °C"
    )

    c3.metric(
        "Average Humidity",
        f"{df['Humidity'].mean():.2f}"
    )

    c4.metric(
        "Average Pressure",
        f"{df['Pressure'].mean():.2f}"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Temperature",
            nbins=35,
            title="Temperature Distribution",
            color="Summary"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            y="Temperature",
            title="Temperature Box Plot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    weather = (
        df["Summary"]
        .value_counts()
        .head(10)
    )

    fig = px.bar(
        x=weather.index,
        y=weather.values,
        labels={
            "x":"Weather Type",
            "y":"Count"
        },
        title="Top Weather Conditions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# Temperature Analytics


elif page == "Temperature Analytics":

    st.header("Temperature Analytics")

    st.dataframe(df.head())

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Minimum",
        f"{df['Temperature'].min():.2f} °C"
    )

    c2.metric(
        "Average",
        f"{df['Temperature'].mean():.2f} °C"
    )

    c3.metric(
        "Maximum",
        f"{df['Temperature'].max():.2f} °C"
    )

    fig = px.line(
        df.head(1000),
        x="Formatted Date",
        y="Temperature",
        title="Temperature Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.histogram(
        df,
        x="Temperature",
        nbins=40,
        title="Temperature Histogram"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.scatter(
        df.sample(3000),
        x="Humidity",
        y="Temperature",
        color="Summary",
        title="Temperature vs Humidity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# Humidity Analytics


elif page == "Humidity Analytics":

    st.header("Humidity Analytics")

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            df,
            x="Humidity",
            nbins=30,
            title="Humidity Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.box(
            df,
            y="Humidity",
            title="Humidity Box Plot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    fig = px.scatter(
        df.sample(3000),
        x="Humidity",
        y="Temperature",
        color="Temperature",
        title="Humidity vs Temperature"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    monthly = df.copy()

    monthly["Month"] = (
        monthly["Formatted Date"]
        .dt.month_name()
    )

    humidity = (
        monthly.groupby("Month")["Humidity"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        humidity,
        x="Month",
        y="Humidity",
        title="Average Monthly Humidity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

   
# WIND ANALYTICS


elif page == "Wind Analytics":

    st.header("🌬 Wind Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Minimum Wind Speed",
        f"{df['WindSpeed'].min():.2f} km/h"
    )

    c2.metric(
        "Average Wind Speed",
        f"{df['WindSpeed'].mean():.2f} km/h"
    )

    c3.metric(
        "Maximum Wind Speed",
        f"{df['WindSpeed'].max():.2f} km/h"
    )

    st.markdown("---")

    fig = px.histogram(
        df,
        x="WindSpeed",
        nbins=40,
        color="Summary",
        title="Wind Speed Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df.sample(min(3000, len(df))),
        x="WindSpeed",
        y="Temperature",
        color="Humidity",
        title="Wind Speed vs Temperature"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        y="WindSpeed",
        title="Wind Speed Box Plot"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Wind Bearing Distribution")

    fig = px.histogram(
        df,
        x="WindBearing",
        nbins=36,
        title="Wind Bearing"
    )

    st.plotly_chart(fig, use_container_width=True)



elif page == "Pressure Analytics":

    st.header("🌪 Pressure Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Minimum Pressure",
        f"{df['Pressure'].min():.2f}"
    )

    c2.metric(
        "Average Pressure",
        f"{df['Pressure'].mean():.2f}"
    )

    c3.metric(
        "Maximum Pressure",
        f"{df['Pressure'].max():.2f}"
    )

    st.markdown("---")

    fig = px.histogram(
        df,
        x="Pressure",
        nbins=40,
        title="Pressure Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df.sample(min(3000, len(df))),
        x="Pressure",
        y="Temperature",
        color="Humidity",
        title="Pressure vs Temperature"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        y="Pressure",
        title="Pressure Box Plot"
    )

    st.plotly_chart(fig, use_container_width=True)



elif page == "Weather Trends":

    st.header("📈 Weather Trends")

    trend_df = df.copy()

    trend_df["Year"] = trend_df["Formatted Date"].dt.year
    trend_df["Month"] = trend_df["Formatted Date"].dt.month

    monthly = (
        trend_df.groupby(["Year", "Month"])
        .agg({
            "Temperature": "mean",
            "Humidity": "mean",
            "WindSpeed": "mean",
            "Pressure": "mean"
        })
        .reset_index()
    )

    monthly["Date"] = pd.to_datetime(
        monthly["Year"].astype(str)
        + "-"
        + monthly["Month"].astype(str)
        + "-01"
    )

    st.subheader("Average Temperature Trend")

    fig = px.line(
        monthly,
        x="Date",
        y="Temperature",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Humidity Trend")

    fig = px.line(
        monthly,
        x="Date",
        y="Humidity",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Wind Speed Trend")

    fig = px.line(
        monthly,
        x="Date",
        y="WindSpeed",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Pressure Trend")

    fig = px.line(
        monthly,
        x="Date",
        y="Pressure",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Weather Summary Count")

    weather_count = (
        df["Summary"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    weather_count.columns = ["Weather", "Count"]

    fig = px.bar(
        weather_count,
        x="Weather",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig, use_container_width=True)


# CORRELATION HEATMAP


    st.markdown("---")

    st.header("🔥 Correlation Heatmap")

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)



    st.header("📊 Advanced Visualizations")

    st.subheader("Temperature vs Pressure")

    fig = px.scatter(
        df.sample(min(3000, len(df))),
        x="Pressure",
        y="Temperature",
        color="Summary",
        size="Humidity"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Temperature vs Wind Speed")

    fig = px.scatter(
        df.sample(min(3000, len(df))),
        x="WindSpeed",
        y="Temperature",
        color="Humidity",
        size="Pressure"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Humidity Distribution by Weather")

    fig = px.box(
        df,
        x="Summary",
        y="Humidity"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pressure Distribution by Weather")

    fig = px.violin(
        df,
        x="Summary",
        y="Pressure",
        box=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weather Summary Pie Chart")

    pie = (
        df["Summary"]
        .value_counts()
        .head(8)
        .reset_index()
    )

    pie.columns = ["Weather", "Count"]

    fig = px.pie(
        pie,
        names="Weather",
        values="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

  
# MACHINE LEARNING


elif page == "Machine Learning":

    st.header("🤖 Temperature Prediction using Multiple Linear Regression")

    features = [
        "Humidity",
        "WindSpeed",
        "WindBearing",
        "Visibility",
        "Pressure"
    ]

    target = "Temperature"

    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    r2 = r2_score(y_test, prediction)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = np.sqrt(mse)

    st.success("Model Trained Successfully ✅")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("R² Score", f"{r2:.4f}")

    c2.metric("MAE", f"{mae:.4f}")

    c3.metric("MSE", f"{mse:.4f}")

    c4.metric("RMSE", f"{rmse:.4f}")

    st.markdown("---")

    st.subheader("Actual vs Predicted Temperature")

    compare = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": prediction
    })

    fig = px.scatter(
        compare,
        x="Actual",
        y="Predicted",
        trendline="ols",
        title="Actual vs Predicted Temperature"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Prediction Form")

    humidity = st.slider(
        "Humidity",
        float(df["Humidity"].min()),
        float(df["Humidity"].max()),
        float(df["Humidity"].mean())
    )

    windspeed = st.number_input(
        "Wind Speed (km/h)",
        value=float(df["WindSpeed"].mean())
    )

    windbearing = st.number_input(
        "Wind Bearing",
        value=float(df["WindBearing"].mean())
    )

    visibility = st.number_input(
        "Visibility (km)",
        value=float(df["Visibility"].mean())
    )

    pressure = st.number_input(
        "Pressure",
        value=float(df["Pressure"].mean())
    )

    if st.button("Predict Temperature"):

        sample = pd.DataFrame({

            "Humidity":[humidity],

            "WindSpeed":[windspeed],

            "WindBearing":[windbearing],

            "Visibility":[visibility],

            "Pressure":[pressure]

        })

        pred = model.predict(sample)

        st.success(
            f"Predicted Temperature : {pred[0]:.2f} °C"
        )

    st.markdown("---")

    st.subheader("Feature Importance")

    importance = pd.DataFrame({

        "Feature":features,

        "Coefficient":model.coef_

    })

    fig = px.bar(

        importance,

        x="Feature",

        y="Coefficient",

        color="Coefficient",

        title="Feature Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Prediction Sample")

    sample_df = compare.head(20)

    st.dataframe(sample_df)

    st.download_button(

        "📥 Download Prediction Report",

        sample_df.to_csv(index=False),

        "prediction_report.csv",

        "text/csv"

    )


# FOOTER


st.markdown("---")

st.markdown(
"""
<center>

### 🌦 Weather Forecast Analytics & Prediction System

Developed using

Python | Pandas | NumPy | Plotly | Matplotlib | Seaborn | Scikit-Learn | Streamlit

Machine Learning Model : **Multiple Linear Regression**

</center>
""",
unsafe_allow_html=True
)