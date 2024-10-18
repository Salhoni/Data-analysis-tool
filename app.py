import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Title of the web app
st.title('Interactive Data Analysis Tool')

# File upload section
uploaded_file = st.file_uploader("Upload a CSV, Excel, or TXT file", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    # Detect the file type and load data accordingly
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, delimiter='\t')
        else:
            st.error("Unsupported file format! Please upload a CSV, Excel, or TXT file.")
        st.success("File uploaded successfully!")

        # Display dataset
        st.subheader('Dataset Preview')
        st.dataframe(df.head())

        # Missing values handling
        st.subheader('Missing Values Handling')
        missing_option = st.selectbox("Select how to handle missing values", ["Drop rows", "Fill with mean", "Fill with median"])
        if missing_option == "Drop rows":
            df_cleaned = df.dropna()
        elif missing_option == "Fill with mean":
            df_cleaned = df.fillna(df.mean())
        elif missing_option == "Fill with median":
            df_cleaned = df.fillna(df.median())

        st.write("Data after handling missing values")
        st.dataframe(df_cleaned.head())

        # Basic Statistics
        st.subheader('Basic Statistics')
        st.write(df_cleaned.describe())

        # Visualization options
        st.subheader('Data Visualization')
        plot_type = st.selectbox("Select plot type", ["Line Plot", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"])
        x_col = st.selectbox("Select X-axis column", df_cleaned.columns)
        y_col = st.selectbox("Select Y-axis column", df_cleaned.columns)

        if plot_type == "Line Plot":
            st.line_chart(df_cleaned[[x_col, y_col]])
        elif plot_type == "Bar Chart":
            st.bar_chart(df_cleaned[[x_col, y_col]])
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df_cleaned, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            fig = px.box(df_cleaned, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif plot_type == "Correlation Heatmap":
            corr = df_cleaned.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
            st.pyplot(fig)

        # Linear Regression model
        st.subheader('Linear Regression Model')
        x_col_reg = st.selectbox("Select feature column (X) for regression", df_cleaned.columns)
        y_col_reg = st.selectbox("Select target column (Y) for regression", df_cleaned.columns)

        X = df_cleaned[[x_col_reg]].values
        y = df_cleaned[y_col_reg].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
        st.write("R-squared:", r2_score(y_test, predictions))

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions)
        ax.plot(y_test, y_test, color='red')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)

        # Advanced Statistical Analysis - T-test
        st.subheader("T-test Analysis")
        ttest_col1 = st.selectbox("Select first column for T-test", df_cleaned.select_dtypes(include=np.number).columns)
        ttest_col2 = st.selectbox("Select second column for T-test", df_cleaned.select_dtypes(include=np.number).columns)
        
        t_stat, p_value = stats.ttest_ind(df_cleaned[ttest_col1].dropna(), df_cleaned[ttest_col2].dropna())
        st.write(f"T-Statistic: {t_stat}")
        st.write(f"P-Value: {p_value}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a file to proceed.")
