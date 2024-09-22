import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Reference the CSV file relative to the app.py file in the repository
csv_file_path = 'Historical returns.csv'

def simulate_portfolio_returns(df, num_years, num_simulations, start_capital, sp_weight, tbond_weight):
    # Clean the 'S&P 500 (includes dividends)' and 'US T. Bond' columns by removing '%' and converting to floats
    df['S&P 500 (includes dividends)'] = df['S&P 500 (includes dividends)'].str.replace('%', '').astype(float) / 100
    df['US T. Bond'] = df['US T. Bond'].str.replace('%', '').astype(float) / 100

    # Extract the S&P 500 and T.Bond returns
    sp500_returns = df['S&P 500 (includes dividends)'].values
    tbond_returns = df['US T. Bond'].values

    # Initialize lists to store final compounded return values, annualized returns, and ending capitals
    final_values = []
    annualized_returns = []
    ending_capitals = []

    # Run the simulations
    for _ in range(num_simulations):
        # Randomly sample returns for the defined number of years for both S&P and T.Bonds
        sampled_sp500_returns = np.random.choice(sp500_returns, size=num_years, replace=True)
        sampled_tbond_returns = np.random.choice(tbond_returns, size=num_years, replace=True)

        # Compute portfolio return: S&P weight * S&P return + T.Bond weight * T.Bond return
        portfolio_returns = sp_weight * sampled_sp500_returns + tbond_weight * sampled_tbond_returns

        # Compute compounded portfolio return: (1 + return1) * (1 + return2) * ... - 1
        compounded_return = np.prod(1 + portfolio_returns) - 1
        final_values.append(compounded_return)

        # Compute the annualized return
        annualized_return = (1 + compounded_return) ** (1 / num_years) - 1
        annualized_returns.append(annualized_return)

        # Compute the ending capital
        ending_capital = start_capital * (1 + compounded_return)
        ending_capitals.append(ending_capital)

    # Convert final compounded returns and annualized returns to percentages for analysis and plotting
    final_values_percent = np.round(np.array(final_values) * 100, 2)
    annualized_returns_percent = np.round(np.array(annualized_returns) * 100, 2)
    ending_capitals = np.round(np.array(ending_capitals), 2)

    # Plot histogram of the final compounded return values in percentages
    fig, ax = plt.subplots()
    ax.hist(final_values_percent, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of Compounded Portfolio Returns ({num_simulations} simulations)')
    ax.set_xlabel('Compounded Return (%)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Plot histogram of the annualized return values in percentages
    fig, ax = plt.subplots()
    ax.hist(annualized_returns_percent, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of Annualized Portfolio Returns ({num_simulations} simulations)')
    ax.set_xlabel('Annualized Return (%)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Plot histogram of the ending capitals
    fig, ax = plt.subplots()
    ax.hist(ending_capitals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of Ending Capitals ({num_simulations} simulations)')
    ax.set_xlabel('Ending Capital')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Calculate percentiles and statistics for compounded returns
    avg_return = round(np.mean(final_values_percent), 2)
    median_return = round(np.median(final_values_percent), 2)
    percentile_25 = round(np.percentile(final_values_percent, 25), 2)
    percentile_75 = round(np.percentile(final_values_percent, 75), 2)
    percentile_10 = round(np.percentile(final_values_percent, 10), 2)
    percentile_90 = round(np.percentile(final_values_percent, 90), 2)
    percentile_5 = round(np.percentile(final_values_percent, 5), 2)
    percentile_95 = round(np.percentile(final_values_percent, 95), 2)

    # Calculate the annualized return based on the median compounded return
    median_annual_return = round(((1 + (median_return / 100)) ** (1 / num_years) - 1) * 100, 2)

    # Calculate percentiles and statistics for annualized returns
    avg_annual_return = round(np.mean(annualized_returns_percent), 2)
    percentile_25_annual = round(np.percentile(annualized_returns_percent, 25), 2)
    percentile_75_annual = round(np.percentile(annualized_returns_percent, 75), 2)
    percentile_10_annual = round(np.percentile(annualized_returns_percent, 10), 2)
    percentile_90_annual = round(np.percentile(annualized_returns_percent, 90), 2)
    percentile_5_annual = round(np.percentile(annualized_returns_percent, 5), 2)
    percentile_95_annual = round(np.percentile(annualized_returns_percent, 95), 2)

    # Calculate percentiles and statistics for ending capital
    avg_ending_capital = round(np.mean(ending_capitals), 2)
    median_ending_capital = round(np.median(ending_capitals), 2)
    percentile_25_capital = round(np.percentile(ending_capitals, 25), 2)
    percentile_75_capital = round(np.percentile(ending_capitals, 75), 2)
    percentile_10_capital = round(np.percentile(ending_capitals, 10), 2)
    percentile_90_capital = round(np.percentile(ending_capitals, 90), 2)
    percentile_5_capital = round(np.percentile(ending_capitals, 5), 2)
    percentile_95_capital = round(np.percentile(ending_capitals, 95), 2)

    # Create a DataFrame to display in a table
    stats_data = {
        'Metric': ['95th Percentile', '90th Percentile', '75th Percentile', 'Average Return', 
                   'Median Return', '25th Percentile', '10th Percentile', '5th Percentile'],
        'Compounded Return (%)': [percentile_95, percentile_90, percentile_75, avg_return, 
                                  median_return, percentile_25, percentile_10, percentile_5],
        'Annualized Return (%)': [percentile_95_annual, percentile_90_annual, percentile_75_annual, avg_annual_return, 
                                  median_annual_return, percentile_25_annual, percentile_10_annual, percentile_5_annual],
        'Ending Capital': [percentile_95_capital, percentile_90_capital, percentile_75_capital, avg_ending_capital, 
                           median_ending_capital, percentile_25_capital, percentile_10_capital, percentile_5_capital]
    }

    stats_df = pd.DataFrame(stats_data)

    # Display the table
    st.table(stats_df)

# Streamlit app
st.title('Portfolio Simulation: S&P 500 and T.Bond')

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Input widgets for number of years, simulations, starting capital, and portfolio weights
num_years = st.number_input('Number of years to simulate', min_value=1, max_value=100, value=30)
num_simulations = st.number_input('Number of simulations', min_value=1, max_value=10000, value=1000)
start_capital = st.number_input('Starting capital', min_value=1, value=10000)
sp_weight = st.number_input('S&P 500 Weight', min_value=0.0, max_value=10.0, value=0.6)
tbond_weight = st.number_input('T.Bond Weight', min_value=0.0, max_value=10.0, value=0.4)

# Run the simulation when the button is clicked
if st.button('Run Simulation'):
    simulate_portfolio_returns(df, num_years, num_simulations, start_capital, sp_weight, tbond_weight)
