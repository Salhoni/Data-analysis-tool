# Import required libraries
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

# Title of the web app
st.title("Interactive Data Analysis Tool")

# Upload dataset section
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt'])

# Load dataset
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.txt'):
            df = pd.read_csv(file, delimiter='\t')
        else:
            st.error("Unsupported file format! Please upload CSV, Excel, or TXT file.")
        return df
    return None

df = load_data(uploaded_file)

if df is not None:
    st.write("### Dataset Preview")
    st.write(df.head())
    
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Handling missing values
    st.sidebar.subheader("Missing Values Handling")
    missing_value_options = st.sidebar.selectbox('Select method to handle missing values', 
                                                 ['None', 'Drop rows', 'Fill with mean', 'Fill with median'])
    
    if missing_value_options == 'Drop rows':
        df_cleaned = df.dropna()
    elif missing_value_options == 'Fill with mean':
        df_cleaned = df.fillna(df.mean())
    elif missing_value_options == 'Fill with median':
        df_cleaned = df.fillna(df.median())
    else:
        df_cleaned = df

    st.write("Updated Data after Missing Values Handling")
    st.write(df_cleaned.head())

    # Visualization Section
    st.sidebar.subheader("Visualize Data")
    plot_type = st.sidebar.selectbox("Select plot type", ['Line Plot', 'Bar Chart', 'Scatter Plot', 'Box Plot', 'Histogram', 'Correlation Heatmap'])
    x_col = st.sidebar.selectbox("X-axis", df_cleaned.columns)
    y_col = st.sidebar.selectbox("Y-axis", df_cleaned.columns)

    # Visualization function
    def visualize_data(plot_type, x_col, y_col):
        if plot_type == "Line Plot":
            plt.figure(figsize=(10, 5))
            plt.plot(df_cleaned[x_col], df_cleaned[y_col], color="green")
            plt.title(f'Line Plot of {y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            st.pyplot(plt)

        elif plot_type == "Bar Chart":
            plt.figure(figsize=(10, 5))
            plt.bar(df_cleaned[x_col], df_cleaned[y_col], color="orange")
            plt.title(f'Bar Chart of {y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            fig = px.scatter(df_cleaned, x=x_col, y=y_col)
            st.plotly_chart(fig)

        elif plot_type == "Box Plot":
            fig = px.box(df_cleaned, x=x_col, y=y_col)
            st.plotly_chart(fig)

        elif plot_type == "Histogram":
            fig = px.histogram(df_cleaned, x=x_col)
            st.plotly_chart(fig)

        elif plot_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm")
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

    visualize_data(plot_type, x_col, y_col)

    # Linear Regression
    st.sidebar.subheader("Linear Regression")
    regression_x_col = st.sidebar.selectbox("Select X for regression", df_cleaned.columns)
    regression_y_col = st.sidebar.selectbox("Select Y for regression", df_cleaned.columns)

    if st.sidebar.button("Run Linear Regression"):
        X = df_cleaned[[regression_x_col]].values.reshape(-1, 1)
        y = df_cleaned[regression_y_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        st.write("### Model Performance")
        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
        st.write("R-squared:", r2_score(y_test, predictions))

        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, predictions)
        plt.plot(y_test, y_test, color='r')
        plt.title('Predictions vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        st.pyplot(plt)

    # Advanced Statistical Analysis
    st.sidebar.subheader("Advanced Statistical Analysis")
    t_test_cols = st.sidebar.multiselect('Select columns for T-Test', df_cleaned.select_dtypes(include=np.number).columns)

    if len(t_test_cols) >= 2:
        col1, col2 = t_test_cols[:2]
        t_stat, p_value = stats.ttest_ind(df_cleaned[col1].dropna(), df_cleaned[col2].dropna())
        st.write(f"T-Statistic: {t_stat}")
        st.write(f"P-Value: {p_value}")
    else:
        st.write("Please select at least 2 columns for T-test.")

else:
    st.write("Please upload a dataset to begin.")

