# Import required libraries
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Number of samples to generate 
n = 1000

# Generate customer ids
customer_ids = np.arange(1, n+1)

# Generate random age, income, and spending_score (1 to 100)
age = np.random.randint(18, 65, n)
income = np.random.randint(20_000, 100_000, n)
spending_score = np.random.randint(1, 100, n)

# Create a DataFrame
df = pd.DataFrame({'CustomerID': customer_ids, 'Age': age, 'Income': income, 'SpendingScore': spending_score})

# Save to CSV
df.to_csv('./raw_data/customer_data.csv', index=False)
