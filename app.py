import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from io import StringIO

st.set_page_config(page_title="Data Analysis Web App", layout="wide", initial_sidebar_state="expanded")

# Set up dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interactive Data Analysis Web App")

# File upload
st.sidebar.title("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
    elif file_type == "txt":
        df = pd.read_csv(uploaded_file, delimiter="\t")
    else:
        st.error("Unsupported file format! Please upload CSV, Excel, or TXT file.")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.dataframe(df.describe())

    # Handling missing values
    missing_options = st.sidebar.selectbox("Handle Missing Values", ["None", "Drop rows", "Fill with mean", "Fill with median"])
    if missing_options == "Drop rows":
        df = df.dropna()
    elif missing_options == "Fill with mean":
        df = df.fillna(df.mean())
    elif missing_options == "Fill with median":
        df = df.fillna(df.median())

    st.write("### Updated Dataset")
    st.dataframe(df.head())

    # Visualization
    st.sidebar.subheader("Visualization Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Line Plot", "Bar Chart", "Scatter Plot", "Box Plot", "Histogram", "Correlation Heatmap"])

    x_col = st.sidebar.selectbox("X-axis", df.columns)
    y_col = st.sidebar.selectbox("Y-axis", df.columns)

    if plot_type == "Line Plot":
        fig, ax = plt.subplots()
        ax.plot(df[x_col], df[y_col], color="green")
        ax.set_title(f'Line Plot of {y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    elif plot_type == "Bar Chart":
        fig, ax = plt.subplots()
        ax.bar(df[x_col], df[y_col], color="orange")
        ax.set_title(f'Bar Chart of {y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif plot_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif plot_type == "Histogram":
        fig = px.histogram(df, x=x_col)
        st.plotly_chart(fig)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Linear Regression Modeling
    st.sidebar.subheader("Linear Regression")
    if st.sidebar.button("Run Linear Regression"):
        X = df[[x_col]].values.reshape(-1, 1)
        y = df[y_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
        st.write(f"R-squared: {r2_score(y_test, predictions)}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions)
        ax.plot(y_test, y_test, color='r')
        ax.set_title('Predictions vs Actual')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        st.pyplot(fig)

    # Advanced Statistical Analysis
    st.sidebar.subheader("Advanced Statistical Analysis")
    stat_test = st.sidebar.selectbox("Select Statistical Test", ["T-Test", "Chi-Square Test", "ANOVA"])

    if stat_test == "T-Test":
        numeric_cols = df.select_dtypes(include=np.number).columns
        col1 = st.sidebar.selectbox("Select First Column for T-Test", numeric_cols)
        col2 = st.sidebar.selectbox("Select Second Column for T-Test", numeric_cols)
        if st.sidebar.button("Run T-Test"):
            t_stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
            st.write(f"T-Statistic: {t_stat}")
            st.write(f"P-Value: {p_value}")

    elif stat_test == "Chi-Square Test":
        cat_cols = df.select_dtypes(include='object').columns
        cat_col1 = st.sidebar.selectbox("Select First Categorical Column", cat_cols)
        cat_col2 = st.sidebar.selectbox("Select Second Categorical Column", cat_cols)
        if st.sidebar.button("Run Chi-Square Test"):
            contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
            chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-Square Statistic: {chi2}")
            st.write(f"P-Value: {p}")

    elif stat_test == "ANOVA":
        cat_col = st.sidebar.selectbox("Select Categorical Column for ANOVA", df.select_dtypes(include='object').columns)
        num_col = st.sidebar.selectbox("Select Numerical Column for ANOVA", df.select_dtypes(include=np.number).columns)
        if st.sidebar.button("Run ANOVA"):
            grouped_data = [group[num_col] for name, group in df.groupby(cat_col)]
            f_stat, p_value = stats.f_oneway(*grouped_data)
            st.write(f"F-Statistic: {f_stat}")
            st.write(f"P-Value: {p_value}")

    # Export results to CSV
    st.sidebar.subheader("Export Data")
    if st.sidebar.button("Export to CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv',
        )
