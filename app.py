# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import os
import io
import joblib
from dateutil import parser
import plotly.express as px
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

# Save the model in memory as bytes and make it downloadable
def save_model_to_memory(model):
    # Save the model to a BytesIO object
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)  # Reset buffer to the beginning
    return buffer

# Function to load yearly hike data
def load_yearly_hike_data(filepath):
    yearly_hike_df = pd.read_csv(filepath)
    return yearly_hike_df

def adjust_for_yearly_hike(result_df, yearly_hike_df):
    # Convert Year column to integer type for consistency
    result_df['Year'] = result_df['Year'].astype(int)
    
    # Merge predictions with hike data
    forecast_df = result_df.merge(yearly_hike_df, on='Year', how='left')
    
    # Calculate cumulative hike factor from 2024 onwards
    forecast_df['Cumulative_Hike_Factor'] = 1  # Initialize as 1 for years up to 2023
    for i, row in forecast_df.iterrows():
        if row['Year'] > 2023:
            # Filter hike data from 2024 to the current year
            cumulative_hikes = yearly_hike_df[(yearly_hike_df['Year'] > 2023) & (yearly_hike_df['Year'] <= row['Year'])]
            cumulative_factor = np.prod(1 + cumulative_hikes['Yearly Hike'].values / 100)
            forecast_df.at[i, 'Cumulative_Hike_Factor'] = cumulative_factor

    # Apply cumulative hike factor to predictions
    forecast_df['Adjusted_BillRate'] = forecast_df['Predicted_BillRate'] * forecast_df['Cumulative_Hike_Factor']
    
    return forecast_df

# Streamlit interface to download the model
def download_model(model):
    buffer = save_model_to_memory(model)
    st.download_button(
        label="Download Model",
        data=buffer,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )

# Function to load and prepare data
def load_data():
    # Load your dataset (assuming it's in CSV format)
    df = pd.read_csv('data.csv')
    
    # Dropping Irrelevant Columns
    #df = df.drop(columns=['SequenceNumber', 'OrderId', 'SkillId', 'FirstName', 'LastName', 'OrderBillRate'])

    df['EffectiveDate'] = df['StartDate'].apply(lambda x: parser.parse(x) if pd.notnull(x) else pd.NaT)
    # Treat 'Year' as numerical to capture trend
    df = df.drop(columns=['StartDate'])
    df['Year'] = df['EffectiveDate'].dt.year
    # scaler = StandardScaler()
    # df['Year'] = scaler.fit_transform(df[['Year_nonscale']])
    # df = df.drop(columns=['Year_nonscale'])
    # Extract the Month from EffectiveDate
    df['Month'] = df['EffectiveDate'].dt.month    
    df['Week'] = df['EffectiveDate'].dt.isocalendar().week
    df['Date'] = df['EffectiveDate'].dt.day
    categorical_cols = ['Region', 'Location', 'Department', 'Skill', 'Month', 'Date', 'Week']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Dropping 'EffectiveDate' as it is no longer needed
    df = df.drop(columns=['EffectiveDate'])
    #df_cleaned = df
    # Removing Outliers Using IQR Method
    Q1 = df['BillRate'].quantile(0.25)  
    Q3 = df['BillRate'].quantile(0.75)  
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df['BillRate'] >= lower_bound) & (df['BillRate'] <= upper_bound)]
    df_cleaned = df_cleaned[(df_cleaned['Diff'] == 1)]
    df_cleaned = df_cleaned[(df_cleaned['Count'] >= 2)]
    df_cleaned =  df_cleaned.drop(columns=['Upper Bound', 'Lower Bound', 'Count', 'Diff','MOnth'])
    st.write(df_cleaned.shape)
    return df_cleaned


def calculate_prediction_intervals(test_df, y_pred, confidence_level=0.95):
    # Calculate residuals
    residuals = test_df['BillRate'] - y_pred
    std_error = np.std(residuals)  # Standard deviation of residuals
    
    # Z-score for desired confidence level (1.96 for 95% confidence)
    z_score = norm.ppf((1 + confidence_level) / 2)
    
    # Calculate upper and lower bounds based on standard error
    margin_of_error = z_score * std_error
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error
    
    return lower_bound, upper_bound

def train_model(df):
    # Convert the categorical columns to 'category' dtype
    categorical_columns = ['Region', 'Location', 'Department', 'Skill', 'Month', 'Date', 'Week']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    train_list = []
    test_list = []

    # Split dataset by 'Week' for training and testing
    for week, group in df.groupby('Week'):
        if len(group) > 4:
            train_group, test_group = train_test_split(group, test_size=0.2, random_state=42)
            train_list.append(train_group)
            test_list.append(test_group)
        else:
            train_list.append(group)

    train_df = pd.concat(train_list, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).reset_index(drop=True)

    if len(test_df) == 0:
        test_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(test_df.index).reset_index(drop=True)

    X_train = train_df.drop(['BillRate','Year','MonthAverage', 'MaxMonth', 'MinMonth', 'StdDevMonth'], axis=1)
    y_train = train_df['BillRate']
    X_test = test_df.drop(['BillRate','Year','MonthAverage', 'MaxMonth', 'MinMonth', 'StdDevMonth'], axis=1)
    y_test = test_df['BillRate']
    
    # Create DMatrix with 'enable_categorical=True'
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    param = {
        'max_depth': 9,              # Allows for complex interactions
        'min_child_weight': 10,       # Helps prevent overfitting on sparse combinations
        'eta': 0.1,                 # Slower learning for better long-term performance
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'nthread': 4,
        'subsample': 0.7,            # Subsampling to prevent overfitting
        'colsample_bytree': 0.8,     # Random feature selection for each tree
        'alpha': 0.1,                # L1 regularization for sparsity
        'lambda': 0.5,               # L2 regularization for complexity control
        'gamma': 0.3,  
        # 'monotone_constraints': "(0, 0, 0, 0,0,0,0,0, 1, 0, 0, 0)",
        'interaction_constraints': [['Month','Location', 'Department','Skill','Region']],
        'tree_method': 'auto',       # Use histogram-based method for faster training
        'enable_categorical': True   # Enable native categorical handling
    }

    # Train the model
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    n_round = 100
    #early_stopping_rounds = 20
    bst = xgb.train(param, dtrain, n_round, evallist)
    
    # Get predictions
    y_pred = bst.predict(dtest)

    lower_bound, upper_bound = calculate_prediction_intervals(test_df, y_pred, confidence_level=0.95)
    
    # MAPE and confidence score
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    confidence_score = 1 - (mape / 100)

    # Result DataFrame
    result_df = test_df.copy()
    result_df['Predicted_BillRate'] = y_pred
    result_df['Lower_Bound'] = lower_bound
    result_df['Upper_Bound'] = upper_bound
    result_df['Confidence_Score'] = confidence_score
    
    return result_df, bst


# Function to calculate MAPE, bounds, and confidence score
def calculate_mape(test_df):
    # Calculate overall MAPE
    test_final_df = adjust_for_yearly_hike(test_df, yearly_hike_df)
    overall_mape = mean_absolute_percentage_error(test_final_df['BillRate'], test_final_df['Predicted_BillRate'])

    # Calculate Confidence Score based on overall MAPE
    test_df['Confidence_Score'] = 1 - (overall_mape / 100)

    # Calculate Weekly MAPE, bounds, and confidence score
    weekly_mape_test = test_df.groupby('Week').apply(
        lambda x: pd.Series({
            'MAPE': mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']),
            'Hires': len(x),
            'Average_Actual_BillRate': x['BillRate'].mean(),
            'Average_Predicted_BillRate': x['Predicted_BillRate'].mean(),
            'Confidence_Score': (1 - (mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))),
            'Lower_Bound': (x['Predicted_BillRate'] * (1 - mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))).mean(),
            'Upper_Bound': (x['Predicted_BillRate'] * (1 + mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))).mean()
        })
    ).reset_index()

    # Calculate Monthly MAPE, bounds, and confidence score
    monthly_mape_test = test_df.groupby('Month').apply(
        lambda x: pd.Series({
            'MAPE': mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']),
            'Hires': len(x),
            'Average_Actual_BillRate': x['BillRate'].mean(),
            'Average_Predicted_BillRate': x['Predicted_BillRate'].mean(),
            'Confidence_Score': (1 - (mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))),
            'Lower_Bound': (x['Predicted_BillRate'] * (1 - mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))).mean(),
            'Upper_Bound': (x['Predicted_BillRate'] * (1 + mean_absolute_percentage_error(x['BillRate'], x['Predicted_BillRate']))).mean()
        })
    ).reset_index()

    # Calculate combined MAPE values
    combined_weekly_mape_test = weekly_mape_test['MAPE'].mean()
    combined_monthly_mape_test = monthly_mape_test['MAPE'].mean()

    return (weekly_mape_test, monthly_mape_test, combined_weekly_mape_test, combined_monthly_mape_test, overall_mape)


import plotly.graph_objects as go

# Function to plot the comparative charts with Plotly
def plot_charts(data, title, time_unit):
    # Convert Bill Rates back to float for plotting
    data['Average_Actual_BillRate'] = data['Average_Actual_BillRate'].str.replace('$', '').astype(float)
    data['Average_Predicted_BillRate'] = data['Average_Predicted_BillRate'].str.replace('$', '').astype(float)

    # Create the figure
    fig = go.Figure()

    # Add Actual Bill Rate trace
    fig.add_trace(go.Scatter(
        x=data[time_unit],
        y=data['Average_Actual_BillRate'],
        mode='lines+markers',
        name='Average Actual Bill Rate',
        marker=dict(color='blue'),
        text=data['Average_Actual_BillRate'],
        hovertemplate='Actual Bill Rate: $%{y:.2f}<br>Week/Month: %{x}<extra></extra>'
    ))

    # Add Predicted Bill Rate trace
    fig.add_trace(go.Scatter(
        x=data[time_unit],
        y=data['Average_Predicted_BillRate'],
        mode='lines+markers',
        name='Average Predicted Bill Rate',
        marker=dict(color='green'),
        text=data['Average_Predicted_BillRate'],
        hovertemplate='Predicted Bill Rate: $%{y:.2f}<br>Week/Month: %{x}<extra></extra>'
    ))

    # Add MAPE trace
    fig.add_trace(go.Scatter(
        x=data[time_unit],
        y=data['MAPE'].str.rstrip('%').astype(float),
        mode='lines+markers',
        name='MAPE (%)',
        marker=dict(color='orange'),
        text=data['MAPE'],
        hovertemplate='MAPE: %{y:.2f}%<br>Week/Month: %{x}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=time_unit.capitalize(),
        yaxis_title='Bill Rate ($)',
        yaxis2=dict(title='MAPE (%)', overlaying='y', side='right'),
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    # Show the figure in Streamlit
    st.plotly_chart(fig)

# Streamlit app layout
st.title("Hallmark POC for Staffing Demand Predictive Analysis")
st.write("Hey there, you have reached the demo page for the Hallmark POC with Colan Infotech Pvt Ltd")
st.write("We received a dataset comprising approximately 50,000 samples from our client. For our modeling process, we utilized an 80:20 split for training and testing our model. We used XGBoost model to predict the Bill rates. Below results are based on evaluation of model on test dataset.")

# Load data and train model
df = load_data()
test_df, model = train_model(df)
yearly_hike_df = load_yearly_hike_data('yearly_hike.csv')



# Tab for Weekly Predictions
weekly_tab, monthly_tab = st.tabs(["Weekly Predictions", "Monthly Predictions"])

with weekly_tab:
    st.subheader("Weekly Predictions")
    
    # Get the weekly MAPE, bounds, and confidence score
    weekly_mape_test, _, combined_weekly_mape_test, _, overall_mape = calculate_mape(test_df)

    # Format Bill Rates to two decimal places
    weekly_mape_test['Average_Actual_BillRate'] = weekly_mape_test['Average_Actual_BillRate'].apply(lambda x: f"${x:.2f}")
    weekly_mape_test['Average_Predicted_BillRate'] = weekly_mape_test['Average_Predicted_BillRate'].apply(lambda x: f"${x:.2f}")

    # Convert MAPE to percentage and add percentage sign
    weekly_mape_test['MAPE'] = (weekly_mape_test['MAPE'] * 100).round(2).astype(str) + '%'
    
    # Format bounds to two decimal places
    weekly_mape_test['Lower_Bound'] = weekly_mape_test['Lower_Bound'].apply(lambda x: f"${x:.2f}")
    weekly_mape_test['Upper_Bound'] = weekly_mape_test['Upper_Bound'].apply(lambda x: f"${x:.2f}")
    
    # Add Confidence Score
    weekly_mape_test['Confidence_Score'] = (weekly_mape_test['Confidence_Score'] * 100).round(2).astype(str) + '%'

    # Display overall MAPE
    st.write(f"Average Weekly MAPE: {combined_weekly_mape_test * 100:.2f}%")
    st.write(f"Overall MAPE: {overall_mape * 100:.2f}%")
    
    # Display the dataframe with all columns, including bounds and confidence score
    st.dataframe(weekly_mape_test)

    # Plotting
    plot_charts(weekly_mape_test, "Weekly MAPE Trend", "Week")

with monthly_tab:
    st.subheader("Monthly Predictions")
    
    # Get the monthly MAPE, bounds, and confidence score
    _, monthly_mape_test, _, combined_monthly_mape_test, _ = calculate_mape(test_df)

    # Format Bill Rates to two decimal places
    monthly_mape_test['Average_Actual_BillRate'] = monthly_mape_test['Average_Actual_BillRate'].apply(lambda x: f"${x:.2f}")
    monthly_mape_test['Average_Predicted_BillRate'] = monthly_mape_test['Average_Predicted_BillRate'].apply(lambda x: f"${x:.2f}")

    # Convert MAPE to percentage and add percentage sign
    monthly_mape_test['MAPE'] = (monthly_mape_test['MAPE'] * 100).round(2).astype(str) + '%'
    
    # Format bounds to two decimal places
    monthly_mape_test['Lower_Bound'] = monthly_mape_test['Lower_Bound'].apply(lambda x: f"${x:.2f}")
    monthly_mape_test['Upper_Bound'] = monthly_mape_test['Upper_Bound'].apply(lambda x: f"${x:.2f}")
    
    # Add Confidence Score
    monthly_mape_test['Confidence_Score'] = (monthly_mape_test['Confidence_Score'] * 100).round(2).astype(str) + '%'

    # Display overall MAPE
    st.write(f"Average Monthly MAPE: {combined_monthly_mape_test * 100:.2f}%")
    st.write(f"Overall MAPE: {overall_mape * 100:.2f}%")
    
    # Display the dataframe with all columns, including bounds and confidence score
    st.dataframe(monthly_mape_test)

    # Plotting
    plot_charts(monthly_mape_test, "Monthly MAPE Trend", "Month")


# Feature Importance Plot
importance = model.get_score(importance_type='weight')
download_model(model)
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
st.pyplot(plt)
