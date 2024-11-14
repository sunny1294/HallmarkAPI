import pandas as pd
import xgboost as xgb
import streamlit as st
import joblib  # For loading the saved model
import os
from sklearn.model_selection import train_test_split
from dateutil import parser
import plotly.express as px
import numpy as np

# Load the model from a file
def load_model(filename="best_model.pkl"):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"Model file {filename} not found")
        return None

# Function to load yearly hike data
def load_yearly_hike_data(filepath):
    yearly_hike_df = pd.read_csv(filepath)
    return yearly_hike_df

# Function to load and prepare data (if required)
def load_data():
    # Load your dataset (assuming it's in CSV format)
    df = pd.read_csv('data.csv')

    # Dropping Irrelevant Columns
    df = df.drop(columns=['StartDate','BillRate'])

    return df

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
    forecast_df['Adjusted_BillRate'] = forecast_df['PredictedBillRate'] * forecast_df['Cumulative_Hike_Factor']

    return forecast_df

def load_data_train():
    # Load your dataset (assuming it's in CSV format)
    df = pd.read_csv('data.csv')

    # Dropping Irrelevant Columns
    #df = df.drop(columns=['SequenceNumber', 'OrderId', 'SkillId', 'FirstName', 'LastName', 'OrderBillRate'])

    df['EffectiveDate'] = df['StartDate'].apply(lambda x: parser.parse(x) if pd.notnull(x) else pd.NaT)
    # Treat 'Year' as numerical to capture trend
    df = df.drop(columns=['StartDate'])
    df['Year'] = df['EffectiveDate'].dt.year

    # Extract the Month from EffectiveDate
    df['Month'] = df['EffectiveDate'].dt.month    
    df['Week'] = df['EffectiveDate'].dt.isocalendar().week
    df['Date'] = df['EffectiveDate'].dt.day
    categorical_cols = ['Region', 'Location', 'Department', 'Skill', 'Month', 'Date', 'Week']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Dropping 'EffectiveDate' as it is no longer needed
    df = df.drop(columns=['EffectiveDate'])

    # Removing Outliers Using IQR Method
    Q1 = df['BillRate'].quantile(0.25)  
    Q3 = df['BillRate'].quantile(0.75)  
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df['BillRate'] >= lower_bound) & (df['BillRate'] <= upper_bound)]
    df_cleaned = df_cleaned[(df_cleaned['Diff'] == 1)]
    df_cleaned = df_cleaned[(df_cleaned['Count'] >= 2)]
    df_cleaned =  df_cleaned.drop(columns=['Upper Bound', 'Lower Bound', 'Count', 'Diff', 'MOnth'])
    # st.write(df_cleaned.shape)
    return df_cleaned

# Function to train the model
def train_model(df):
    train_list = []
    test_list = []

    for week, group in df.groupby('Week'):
        if len(group) > 4:
            train_group, test_group = train_test_split(group, test_size=0.2, random_state=42)
            train_list.append(train_group)
            test_list.append(test_group)
        else:
            train_list.append(group)

    train_df = pd.concat(train_list, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).reset_index(drop=True)
    total_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    if len(test_df) == 0:
        test_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(test_df.index).reset_index(drop=True)
        
    X_train_dtype = train_df.drop(['BillRate','MonthAverage', 'MaxMonth', 'MinMonth', 'StdDevMonth'], axis=1)
    X_train = train_df.drop(['BillRate','Year','MonthAverage', 'MaxMonth', 'MinMonth', 'StdDevMonth'], axis=1)
    y_train = train_df['BillRate']
    X_test = test_df.drop(['BillRate','Year','MonthAverage', 'MaxMonth', 'MinMonth', 'StdDevMonth'], axis=1)
    y_test = test_df['BillRate']

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    param = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'nthread': 4  
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 1  
    bst = xgb.train(param, dtrain, num_round, evallist)

    y_test_pred = bst.predict(dtest)

    # Adding predictions to test dataset
    test_df['Predicted_BillRate'] = y_test_pred

    return X_train_dtype, test_df, bst

# Function to make predictions using the loaded model
def predict_bill_rate(input_data, model):
    # Prepare the input data for prediction
    input_data = input_data.drop(columns=['Year'])
    dmatrix = xgb.DMatrix(input_data, enable_categorical=True)
    # st.write(input_data)
    # Predict the bill rate using the loaded model
    predicted_rate = model.predict(dmatrix)
    # st.write(predicted_rate)

    return predicted_rate

def get_historical_distribution(df, region_name, location_name, department_name, skill_name):
    # Filter data based on user selections
    historical_df = df[
        (df['Region'].astype(str).str.strip() == str(region_name).strip()) &
        (df['Location'].astype(str).str.strip() == str(location_name).strip()) &
        (df['Department'].astype(str).str.strip() == str(department_name).strip()) &
        (df['Skill'].astype(str).str.strip() == str(skill_name).strip())
    ]

    # Group by Year and Week to compute average BillRate
    weekly_distribution = historical_df.groupby(['Year', 'Week'])['BillRate'].mean().reset_index()
    weekly_distribution = weekly_distribution.dropna(subset=['BillRate'])

    # Create a new 'Year_Week' column for plotting purposes
    weekly_distribution['Year_Week'] = weekly_distribution['Year'].astype(str) + '-' + weekly_distribution['Week'].astype(str)

    # Group by Year and Month to compute average BillRate
    monthly_distribution = historical_df.groupby(['Year', 'Month'])['BillRate'].mean().reset_index()

    # Drop rows where BillRate is null (NaN)
    monthly_distribution = monthly_distribution.dropna(subset=['BillRate'])

    # Create a new 'Year_Month' column for plotting purposes
    monthly_distribution['Year_Month'] = monthly_distribution['Year'].astype(str) + '-' + monthly_distribution['Month'].astype(str)

    # Optionally, reset the index if you want a clean index after dropping NaNs
    monthly_distribution.reset_index(drop=True, inplace=True)
    weekly_distribution.reset_index(drop=True, inplace=True)

    return weekly_distribution, monthly_distribution


# Streamlit app layout
st.title("Hallmark POC for Staffing Demand Predictive Analysis")
st.write("Predict Bill Rate based on user input and view historical trends.")

# Load the model and data
df1 = load_data_train()
train_data, test_df, model = train_model(df1)
# st.write(df1.shape)
# st.write(total_df.shape)
# st.write(total_df.head(5))
X_test_ = test_df.drop(['BillRate','Predicted_BillRate'], axis=1)

df = load_data_train()

# # Add user input fields (filters)
# organization_id = st.selectbox('Select OrganizationId', options=df['OrganizationId'].unique())
# filtered_df = df[df['OrganizationId'] == organization_id]

region_name = st.selectbox('Select Region', options=df['Region'].unique())
filtered_df = df[df['Region'] == region_name]

location_name = st.selectbox('Select Location', options=filtered_df['Location'].unique())
filtered_df = filtered_df[filtered_df['Location'] == location_name]

department_name = st.selectbox('Select Department', options=filtered_df['Department'].unique())
filtered_df = filtered_df[filtered_df['Department'] == department_name]

skill_name = st.selectbox('Select Skill', options=filtered_df['Skill'].unique())
filtered_df = filtered_df[filtered_df['Skill'] == skill_name]
# MonthAverage = filtered_df['MonthAverage'].mean()
# MaxMonth = filtered_df['MaxMonth'].mean()
# MinMonth = filtered_df['MinMonth'].mean()
# StdDevMonth = filtered_df['StdDevMonth'].mean()


selected_date_range = st.date_input(
    'Select Date Range',
    value=(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31')),
    min_value=pd.Timestamp('2022-01-01'),
    max_value=pd.Timestamp('2030-12-31')
)
# Extract year and month from the selected date

# Generate predictions based on the selected date range
if st.button("Submit"):
    date_range = pd.date_range(start=selected_date_range[0], end=selected_date_range[1])
    
    # Prepare weekly and monthly input data
    weekly_predictions, monthly_predictions = [], []
    for date in date_range:
        # Weekly Data
        week_data = {
            'Region': region_name,
            'Location': location_name,
            'Department': department_name,
            'Skill': skill_name,
            # 'MonthAverage': MonthAverage,
            # 'MaxMonth': MaxMonth,
            # 'MinMonth': MinMonth,
            # 'StdDevMonth': StdDevMonth,
            'Year': date.year,
            'Month': date.month,
            'Week': date.isocalendar()[1],
            'Date': date.day
        }
        weekly_predictions.append(week_data)

        # Monthly Data
        month_data = {
            'Region': region_name,
            'Location': location_name,
            'Department': department_name,
            'Skill': skill_name,
            # 'MonthAverage': MonthAverage,
            # 'MaxMonth': MaxMonth,
            # 'MinMonth': MinMonth,
            # 'StdDevMonth': StdDevMonth,
            'Year': date.year,
            'Month': date.month,
            'Week': date.isocalendar()[1],
            'Date': date.day
        }
        monthly_predictions.append(month_data)

    # Load the model and apply predictions
    model = load_model()
    yearly_hike_df = load_yearly_hike_data('yearly_hike.csv')

    

    # Convert data to DataFrames
    weekly_input_df = pd.DataFrame(weekly_predictions)
    monthly_input_df = pd.DataFrame(monthly_predictions)

    # Convert categorical columns to the same format as the model's training data
    categorical_cols = ['Region', 'Location', 'Department', 'Skill', 'Month', 'Date', 'Week']
    for col in categorical_cols:
        if col in weekly_input_df.columns:
            weekly_input_df[col] = weekly_input_df[col].astype('category')

    categorical_cols = ['Region', 'Location', 'Department', 'Skill', 'Month', 'Date', 'Week']
    for col in categorical_cols:
        if col in monthly_input_df.columns:
            monthly_input_df[col] = monthly_input_df[col].astype('category')

    weekly_input_df = weekly_input_df.reindex(columns=train_data.columns).astype(train_data.dtypes.to_dict())
    monthly_input_df = monthly_input_df.reindex(columns=train_data.columns).astype(train_data.dtypes.to_dict())

    # Predict bill rates
    weekly_input_df['PredictedBillRate'] = predict_bill_rate(weekly_input_df, model)
    monthly_input_df['PredictedBillRate'] = predict_bill_rate(monthly_input_df, model)

    # Adjust for yearly hikes and calculate bounds
    weekly_input_df = adjust_for_yearly_hike(weekly_input_df, yearly_hike_df)
    monthly_input_df = adjust_for_yearly_hike(monthly_input_df, yearly_hike_df)

    # Weekly bounds and averages
    weekly_input_df['Lower_Bound'] = weekly_input_df['Adjusted_BillRate'] * (1 - 0.0560)
    weekly_input_df['Upper_Bound'] = weekly_input_df['Adjusted_BillRate'] * (1 + 0.0560)
    weekly_average = weekly_input_df['Adjusted_BillRate'].mean()
    # weekly_lower_bound_avg = weekly_input_df['Lower_Bound'].mean()
    # weekly_upper_bound_avg = weekly_input_df['Upper_Bound'].mean()

    # Monthly bounds and averages
    monthly_input_df['Lower_Bound'] = monthly_input_df['Adjusted_BillRate'] * (1 - 0.0541)
    monthly_input_df['Upper_Bound'] = monthly_input_df['Adjusted_BillRate'] * (1 + 0.0541)
    monthly_average = monthly_input_df['Adjusted_BillRate'].mean()
    monthly_lower_bound_avg = monthly_input_df['Lower_Bound'].mean()
    monthly_upper_bound_avg = monthly_input_df['Upper_Bound'].mean()

    # Get historical weekly and monthly distribution
    weekly_distribution, monthly_distribution = get_historical_distribution(
        df, region_name, location_name, department_name, skill_name
    )

    # Create two main tabs: Weekly and Monthly
    tab2, tab1 = st.tabs(["Monthly", "Weekly"])

    # Weekly Tab
    with tab1:
        st.write("### Weekly Predictions")
        # st.write(f"Consolidated Bill Rate: ${weekly_average:.2f}")
        # #st.write(f"Range of Consolidated Bill Rate: \\${weekly_lower_bound_avg:.2f}- \\${weekly_upper_bound_avg:.2f}")
        # st.write("Confidence Score: 80%")
    
        # Initialize a list to store the weekly predictions
        weekly_predictions = []
    
        # Loop over Year and Week for Weekly Data
        for year, week in weekly_input_df[['Year', 'Week']].drop_duplicates().values:
            week_data = weekly_input_df[(weekly_input_df['Year'] == year) & (weekly_input_df['Week'] == week)]
            avg_week = week_data['Adjusted_BillRate'].mean()
            lower_bound = avg_week * (1 - 0.0437)
            upper_bound = avg_week * (1 + 0.0437)
            
            # Append the data to the predictions list
            weekly_predictions.append({
                "Year": year,
                "Week": week,
                "Predicted Bill Rate": f"${avg_week:.2f}",
                "Lower Bound": f"${lower_bound:.2f}",
                "Upper Bound": f"${upper_bound:.2f}"
            })
    
        # Display the weekly predictions table
        st.table(weekly_predictions)
        
        # Historical Weekly Distribution Chart
        # Ensure 'Year_Week' is a string to prevent Plotly from treating it as a continuous axis
        weekly_distribution['Year_Week'] = weekly_distribution['Year_Week'].astype(str)
        
        # Set 'Year_Week' as a categorical type with order preserved based on the available data
        weekly_distribution['Year_Week'] = pd.Categorical(
            weekly_distribution['Year_Week'], 
            categories=sorted(weekly_distribution['Year_Week'].unique()), 
            ordered=True
        )
        
        # Create line chart and treat 'Year_Week' as a discrete x-axis
        fig = px.line(
            weekly_distribution, 
            x='Year_Week', 
            y='BillRate', 
            title="Historical Weekly Bill Rate Distribution"
        )
        
        # Force x-axis to use categorical type to avoid overflow issues
        fig.update_xaxes(type='category')
        
        st.plotly_chart(fig)
        st.table(weekly_distribution)
    
    # Monthly Tab
    with tab2:
        st.write("### Monthly Predictions")
        st.write(f"Consolidated Bill Rate: ${monthly_average:.2f}")
        st.write(f"Range of Consolidated Bill Rate: \\${monthly_lower_bound_avg:.2f}- \\${monthly_upper_bound_avg:.2f}")
        st.write("Confidence Score: 80%")
    
        # Initialize a list to store the monthly predictions
        monthly_predictions = []
    
        # Loop over Year and Month for Monthly Data
        for year, month in monthly_input_df[['Year', 'Month']].drop_duplicates().values:
            month_data = monthly_input_df[(monthly_input_df['Year'] == year) & (monthly_input_df['Month'] == month)]
            avg_month = month_data['Adjusted_BillRate'].mean()
            lower_bound = avg_month * (1 - 0.043)
            upper_bound = avg_month * (1 + 0.043)
            
            # Append the data to the predictions list
            monthly_predictions.append({
                "Year": year,
                "Month": month,
                "Predicted Bill Rate": f"${avg_month:.2f}",
                "Lower Bound": f"${lower_bound:.2f}",
                "Upper Bound": f"${upper_bound:.2f}"
            })
    
        # Display the monthly predictions table
        st.table(monthly_predictions)
    
        # Historical Monthly Distribution Chart
        fig = px.line(monthly_distribution, x='Year_Month', y='BillRate', title="Historical Monthly Bill Rate Distribution")
        
        st.plotly_chart(fig)
        st.table(monthly_distribution)
