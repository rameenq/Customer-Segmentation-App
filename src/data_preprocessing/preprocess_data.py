# Importing the required libraries
# pandas is being used for data manipulation and StandardScaler is for feature scaling
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Defining a function to load data from a CSV file into a Pandas DataFrame
def load_data(filepath):
    return pd.read_csv(filepath)

# Defining a function to scale specified features in a DataFrame
# StandardScaler standardizes the features by removing the mean and scaling to unit variance
def scale_features(df, features):
    # Initializing the StandardScaler object
    scaler = StandardScaler()
    
    # Scaling the features in-place and update the DataFrame
    df[features] = scaler.fit_transform(df[features])
    
    # Returning the updated DataFrame
    return df

# Defining a function to save a DataFrame to a CSV file
def save_data(df, filepath):
    # Saving DataFrame to CSV, without the index column
    df.to_csv(filepath, index=False)

# Checking if the script is being run as the main program
if __name__ == "__main__":
    # Specify the relative file paths for the input and output CSV files
    input_filepath = '../../data/raw_data/customer_data.csv'
    output_filepath = '../../data/processed_data/segmented_customers.csv'
    
    # Loading data from the input CSV file into a DataFrame
    df = load_data(input_filepath)
    
    # Scaling the 'Age', 'Income', and 'SpendingScore' features
    df = scale_features(df, ['Age', 'Income', 'SpendingScore'])
    
    # Saving the processed data to the output CSV file
    save_data(df, output_filepath)
