import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define the file path to your CSV file
csv_file_path = 'C:/Users/benja/OneDrive/Desktop/Kelly Sheets/git/Historical returns.csv'

def simulate_sp500_compounded_returns(df, num_years, num_simulations):
    # Clean the 'S&P 500 (includes dividends)' column by removing '%' and converting to floats
    df['S&P 500 (includes dividends)'] = df['S&P 500 (includes dividends)'].str.replace('%', '').astype(float) / 100
    
    # Extract the S&P 500 returns
    sp500_returns = df['S&P 500 (includes dividends)'].values

    # Initialize an empty list to store the final compounded return values from each simulation
    final_values = []

    # Run the simulations
    for _ in range(num_simulations):
        # Randomly sample returns for the defined number of years
        sampled_returns = np.random.choice(sp500_returns, size=num_years, replace=True)

        # Compute compounded return: (1 + return1) * (1 + return2) * ... - 1
        compounded_return = np.prod(1 + sampled_returns) - 1
        
        # Store the final compounded return
        final_values.append(compounded_return)

    # Convert final compounded returns to percentages for analysis and plotting
    final_values_percent = np.array(final_values) * 100

    # Plot histogram of the final compounded return values in percentages
    fig, ax = plt.subplots()
    ax.hist(final_values_percent, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of Compounded S&P 500 Returns ({num_simulations} simulations)')
    ax.set_xlabel('Compounded Return (%)')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)

    # Calculate percentiles and statistics
    avg_return = np.mean(final_values_percent)
    median_return = np.median(final_values_percent)
    percentile_25 = np.percentile(final_values_percent, 25)
    percentile_75 = np.percentile(final_values_percent, 75)
    percentile_10 = np.percentile(final_values_percent, 10)
    percentile_90 = np.percentile(final_values_percent, 90)
    percentile_5 = np.percentile(final_values_percent, 5)
    percentile_95 = np.percentile(final_values_percent, 95)

    # Create a DataFrame to display in a table, reordered as requested
    stats_data = {
        'Metric': ['95th Percentile', '90th Percentile', '75th Percentile', 'Average Return', 
                   'Median Return', '25th Percentile', '10th Percentile', '5th Percentile'],
        'Value (%)': [percentile_95, percentile_90, percentile_75, avg_return, 
                      median_return, percentile_25, percentile_10, percentile_5]
    }

    stats_df = pd.DataFrame(stats_data)

    # Display the table
    st.table(stats_df)

# Streamlit app
st.title('S&P 500 Compounded Returns Simulation')

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Input widgets for number of years and simulations
num_years = st.number_input('Number of years to simulate', min_value=1, max_value=100, value=10)
num_simulations = st.number_input('Number of simulations', min_value=1, max_value=10000, value=100)

# Run the simulation when the button is clicked
if st.button('Run Simulation'):
    simulate_sp500_compounded_returns(df, num_years, num_simulations)
