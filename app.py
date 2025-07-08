import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
# -----------------------------------
# Helper functions


def adjusted_r2(r2, n, p):
    # Adjusted R-squared formula
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

# -----------------------------------
# EDA section

def eda_app(df):
    st.header('Exploratory Data Analysis (EDA)')
    st.subheader("📋 Dataset Preview")
    st.dataframe(df)



    #st.markdown('**Data Description:**')
    st.subheader("📊 Data Summary")
    st.write(df.describe())

    st.subheader("🧩 Missing Values")
    st.write(df.isnull().sum())
    # 🧹 Handle missing data
    st.subheader("🧹 Handle Missing Data")
    if st.checkbox("Fill missing values"):
        method = st.radio("Choose method", ["Mean", "Median", "Mode (for all)"])
        if method == "Mean":
            for col in df.select_dtypes(include='number').columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif method == "Median":
            for col in df.select_dtypes(include='number').columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif method == "Mode (for all)":
            for col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
    elif st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)

    st.write("✅ Missing values handled. Here's the updated dataset:")
    st.dataframe(df)

    st.subheader("Box Plot (for outlier detection)")
    selected_col_box = st.selectbox("Select a column to plot distribution", df.columns,key="column_explore")
    if pd.api.types.is_numeric_dtype(df[selected_col_box]):
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[selected_col_box], ax=ax2)
        st.pyplot(fig2)



    # -----------------------------------
    # 📉 Outlier Handling Section
    st.subheader("📉 Outlier Handling")

    if st.checkbox("Handle Outliers in numeric columns"):
       # method = st.radio("Choose method for outlier detection", ["IQR", "Z-Score"])

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            original_len = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = original_len - len(df)
            if removed > 0:
                st.write(f"Removed {removed} outliers from `{col}` using IQR.")


        st.write("✅ Outliers removed. Here's the updated dataset:")
        st.dataframe(df)


    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Plot Distributions")
    selected_col_dist = st.selectbox("Select a column to plot distribution", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_col_dist]):
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col_dist], kde=True, ax=ax)
        st.pyplot(fig)

    else:
        st.write("Selected column is non-numeric, skipping histogram.")

# ✅ Column-Level Exploration
    st.subheader("📌 Column-Level Exploration")
    selected_col = st.selectbox("Select a column to explore", df.columns, key="col_explore")

    st.write(f"**Data Type:** {df[selected_col].dtype}")
    st.write(f"**Unique Values:** {df[selected_col].nunique()}")
    st.write("**Summary Statistics:**")
    st.write(df[selected_col].describe())

    if pd.api.types.is_object_dtype(df[selected_col]) or pd.api.types.is_categorical_dtype(df[selected_col]):
        st.write("**Value Counts:**")
        st.write(df[selected_col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(y=df[selected_col], order=df[selected_col].value_counts().index, ax=ax)
        st.pyplot(fig)

    elif pd.api.types.is_numeric_dtype(df[selected_col]):
        st.write("**Histogram with KDE:**")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("**Box Plot (for outlier detection):**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax2)
        st.pyplot(fig2)

    else:
        st.warning("Selected column is neither numeric nor categorical.")


    # 📊 Advanced Visualizations
    st.subheader("📊 Advanced Visualizations")

    # Scatter Plot
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        st.markdown("**Scatter Plot**")
        x_col = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
        y_col = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax3)
        st.pyplot(fig3)

    # Pair Plot
    if st.checkbox("Show Pair Plot for numeric columns"):
        selected_pair_cols = st.multiselect("Select numeric columns for pairplot", numeric_cols, default=numeric_cols[:3])
        if len(selected_pair_cols) >= 2:
            pair_fig = sns.pairplot(df[selected_pair_cols])
            st.pyplot(pair_fig)



        # Pie Chart for Categorical Column
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()



    if len(cat_cols) > 0:
        st.markdown("**Pie Chart for a Categorical Column**")
        pie_col = st.selectbox("Select a categorical column", cat_cols, key="bar_col")
        bar_data = df[pie_col].value_counts().reset_index()
        bar_data.columns = [pie_col, 'Count']

        # Create bar chart
        fig, ax = plt.subplots()
        sns.barplot(data=bar_data, x=pie_col, y='Count', ax=ax, palette='Set2')

        # Optional: rotate labels, add title
        ax.set_title(f"Distribution of {pie_col}")
        ax.set_ylabel("Count")
        ax.set_xlabel(pie_col)
        for container in ax.containers:
            ax.bar_label(container, label_type='edge', fontsize=10)
        # show count labels on bars

        st.pyplot(fig)




    # 📁 Data Export
    st.subheader("📁 Export Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='cleaned_dataset.csv',
        mime='text/csv'
            )



# -----------------------------------
# ML model section
def build_model(df, split_size, seed_number):
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.linear_model import BayesianRidge, HuberRegressor, PassiveAggressiveRegressor
    from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

    st.header('Machine Learning Model Comparison')
    df = df.copy()

    target_column = st.selectbox("Select the target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])

    if target_column and len(feature_columns) > 0:
        X = df[feature_columns]
        Y = df[target_column]
        p = X.shape[1]

        st.subheader("Dataset dimension")
        st.write('X shape:', X.shape)
        st.write('Y shape:', Y.shape)

        st.subheader("Variable details")
        st.write('X variables :', list(X.columns[:20]))
        st.write('Y variable:', Y.name)

        # Train/test split
        test_size = 1 - split_size / 100
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=seed_number
        )

        # ✅ Extended model list
        all_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet": ElasticNet(),
            "Bayesian Ridge": BayesianRidge(),
            "Huber Regressor": HuberRegressor(),
            "Passive Aggressive": PassiveAggressiveRegressor(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=seed_number),
            "Random Forest": RandomForestRegressor(random_state=seed_number),
            "Gradient Boosting": GradientBoostingRegressor(random_state=seed_number),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=seed_number),
            "Bagging Regressor": BaggingRegressor(random_state=seed_number),
            "Support Vector Regressor": SVR(),
            "XGBoost": XGBRegressor(verbosity=0),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor(verbose=0)
        }

        # ✅ Default models
        default_model_names = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Support Vector Regressor",

            "XGBoost"
        ]

        # ✅ User-selectable model list
        st.subheader("📌 Select Regression Models to Compare")
        selected_model_names = st.multiselect(
            "Choose regression models to compare:",
            options=list(all_models.keys()),
            default=default_model_names
        )

        # ✅ Fallback to default if none selected
        if not selected_model_names:
            st.info("No models selected. Using default models.")
            selected_model_names = default_model_names

        models = {name: all_models[name] for name in selected_model_names}

        # 🧪 Train + Evaluate models
        st.subheader(" Model performance")

        results = []

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

                r2 = r2_score(Y_test, Y_pred)
                adj_r2 = adjusted_r2(r2, len(Y_test), p)
                mae = mean_absolute_error(Y_test, Y_pred)
                mse = mean_squared_error(Y_test, Y_pred)
                rmse = np.sqrt(mse)

                results.append({
                    'Model': name,
                    'R² Score': r2,
                    'Adjusted R²': adj_r2 if adj_r2 is not None else np.nan,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse
                })
            except Exception as e:
                st.warning(f"{name} failed: {e}")

        results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False).reset_index(drop=True)
        st.dataframe(results_df.round(3))

        st.subheader("Performance Metrics Graphs")

        fig_r2 = px.bar(results_df, x='Model', y='R² Score', title="R² Score by Model", color='Model', range_y=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)

        fig_adj_r2 = px.bar(results_df, x='Model', y='Adjusted R²', title="Adjusted R² Score by Model", color='Model',
                            range_y=[0, 1])
        st.plotly_chart(fig_adj_r2, use_container_width=True)

        fig_rmse = px.bar(results_df, x='Model', y='RMSE', title="RMSE by Model", color='Model')
        st.plotly_chart(fig_rmse, use_container_width=True)

    else:
        st.warning("Please select a target column and at least one feature column.")

# Main app

#st.set_page_config(page_title='EDA + ML Algorithm Comparison App', layout='wide')
# Page title
st.markdown('''
# **Combined EDA and Machine Learning Algorithm Comparison App**


---
''')
#st.title('Combined EDA and Machine Learning Algorithm Comparison App')

with st.sidebar:
    st.header('Upload your dataset (CSV)')
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv','xlsx','xls'])

    st.header('ML Parameters')
    split_size = st.slider('Training data split percentage', 10, 90, 80, 5)
    seed_number = st.slider('Random seed number', 1, 100, 42, 1)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(uploaded_file)
    elif file_extension == '.csv':
        df = pd.read_csv(uploaded_file)
    else:
        st.error('Unsupported file format! Please upload a CSV or Excel file.')
        st.stop()

    eda_app(df)

    #st.header('Machine Learning Model Comparison')
    build_model(df, split_size, seed_number)


else:
    st.info('Awaiting CSV file upload to begin analysis.')
