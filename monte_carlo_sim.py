import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Reference the CSV file relative to the app.py file in the repository
csv_file_path = 'Historical returns.csv'

def simulate_portfolio_returns(df, num_years, num_simulations, start_capital, sp_weight, tbond_weight, use_inflation_adjusted, interest_basis):
    # Select columns based on user preference for inflation-adjusted returns
    if use_inflation_adjusted:
        sp500_col = 'S&P 500 (includes dividends) inflation adjusted'
        tbond_col = 'US T. Bond inflation adjusted'
        tbill_col = '3-month T.Bill inflation adjusted'
    else:
        sp500_col = 'S&P 500 (includes dividends)'
        tbond_col = 'US T. Bond'
        tbill_col = '3-month T.Bill'

    # Clean the selected columns by removing '%' and converting to floats
    df[sp500_col] = df[sp500_col].str.replace('%', '').astype(float) / 100
    df[tbond_col] = df[tbond_col].str.replace('%', '').astype(float) / 100
    df[tbill_col] = df[tbill_col].str.replace('%', '').astype(float) / 100

    # Extract the selected S&P 500, T.Bond, and 3-month T.Bill returns
    sp500_returns = df[sp500_col].values
    tbond_returns = df[tbond_col].values
    tbill_returns = df[tbill_col].values

    # Initialize lists to store final compounded return values and ending capitals
    final_values = []
    ending_capitals = []

    # Run the simulations
    for _ in range(num_simulations):
        # Randomly sample indices for the defined number of years (same index for both S&P and T.Bonds)
        sampled_indices = np.random.choice(len(sp500_returns), size=num_years, replace=True)
        
        # Use the sampled indices to select the corresponding returns for both S&P 500, T.Bonds, and 3-month T.Bill
        sampled_sp500_returns = sp500_returns[sampled_indices]
        sampled_tbond_returns = tbond_returns[sampled_indices]
        sampled_tbill_returns = tbill_returns[sampled_indices]

        # Compute portfolio return: S&P weight * S&P return + T.Bond weight * T.Bond return
        portfolio_returns = sp_weight * sampled_sp500_returns + tbond_weight * sampled_tbond_returns

        # Calculate total allocation and check if it's above 1
        total_allocation = sp_weight + tbond_weight
        if total_allocation > 1:
            # Apply the negative return for the excess allocation with interest basis
            excess_allocation = total_allocation - 1
            interest_cost = excess_allocation * (sampled_tbill_returns + interest_basis) * -1
            portfolio_returns += interest_cost

        # Compute compounded portfolio return: (1 + return1) * (1 + return2) * ... - 1
        compounded_return = np.prod(1 + portfolio_returns) - 1
        final_values.append(compounded_return)

        # Compute the ending capital
        ending_capital = start_capital * (1 + compounded_return)
        ending_capitals.append(ending_capital)

    # Convert final compounded returns to percentages for analysis and plotting
    final_values_percent = np.round(np.array(final_values) * 100, 2)
    ending_capitals = np.round(np.array(ending_capitals), 2)

    # Calculate percentiles and statistics for compounded returns
    avg_return = round(np.mean(final_values_percent), 2)
    median_return = round(np.median(final_values_percent), 2)
    percentile_25 = round(np.percentile(final_values_percent, 25), 2)
    percentile_75 = round(np.percentile(final_values_percent, 75), 2)
    percentile_10 = round(np.percentile(final_values_percent, 10), 2)
    percentile_90 = round(np.percentile(final_values_percent, 90), 2)
    percentile_5 = round(np.percentile(final_values_percent, 5), 2)
    percentile_95 = round(np.percentile(final_values_percent, 95), 2)

    # Calculate annualized returns based on the compounded returns for each percentile
    avg_annual_return = round(((avg_return / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    median_annual_return = round(((median_return / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_25_annual = round(((percentile_25 / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_75_annual = round(((percentile_75 / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_10_annual = round(((percentile_10 / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_90_annual = round(((percentile_90 / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_5_annual = round(((percentile_5 / 100) + 1) ** (1 / num_years) - 1, 4) * 100
    percentile_95_annual = round(((percentile_95 / 100) + 1) ** (1 / num_years) - 1, 4) * 100

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

    # Convert DataFrame to display rounded values using formatting
    stats_df = pd.DataFrame(stats_data).style.format({
        'Compounded Return (%)': '{:.2f}',
        'Annualized Return (%)': '{:.2f}',
        'Ending Capital': '{:,.2f}'  # This will add commas to the 'Ending Capital' column
    })

    return stats_df, final_values_percent, ending_capitals

def optimize_portfolio(df, num_years, num_simulations, start_capital, use_inflation_adjusted, interest_basis, sp_weight_range, tbond_weight_range):
    best_25th_percentile = -np.inf
    best_sp_weight = 0
    best_tbond_weight = 0

    # Grid search over S&P and T.Bond weight combinations
    for sp_weight in sp_weight_range:
        for tbond_weight in tbond_weight_range:
            if sp_weight + tbond_weight <= 1:  # Ensure weights sum <= 1
                percentile_25, _, _, _ = simulate_portfolio_returns(df, num_years, num_simulations, start_capital, sp_weight, tbond_weight, use_inflation_adjusted, interest_basis)
                if percentile_25 > best_25th_percentile:
                    best_25th_percentile = percentile_25
                    best_sp_weight = sp_weight
                    best_tbond_weight = tbond_weight

    return best_25th_percentile, best_sp_weight, best_tbond_weight

# Streamlit app
st.title('Portfolio Simulation: S&P 500 and T.Bond')

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

use_inflation_adjusted = st.checkbox("Use inflation-adjusted returns", value=False)

# Input widgets for number of years, simulations, starting capital, portfolio weights, and interest basis
num_years = st.number_input('Number of years to simulate', min_value=1, max_value=1000, value=30)
num_simulations = st.number_input('Number of simulations', min_value=1, max_value=100000, value=10000)
start_capital = st.number_input('Starting capital', min_value=1, value=10000)
interest_basis = st.number_input('Interest basis (added to T-bill rate)', min_value=-1.0, max_value=1.0, value=0.0, step =0.001)

# Optimization toggle
optimize = st.checkbox('Optimize for best 25th percentile')

if optimize:
    # Define ranges for S&P and T.Bond weights
    sp_weight_min = st.number_input('S&P 500 Weight Min', min_value=0.0, max_value=5.0, value=0.0)
    sp_weight_max = st.number_input('S&P 500 Weight Max', min_value=0.0, max_value=5.0, value=1.0)
    sp_weight_step = st.number_input('S&P 500 Weight Step', min_value=0.01, max_value=5.0, value=0.1)
    
    tbond_weight_min = st.number_input('T.Bond Weight Min', min_value=0.0, max_value=5.0, value=0.0)
    tbond_weight_max = st.number_input('T.Bond Weight Max', min_value=0.0, max_value=5.0, value=1.0)
    tbond_weight_step = st.number_input('T.Bond Weight Step', min_value=0.01, max_value=5.0, value=0.1)

    sp_weight_range = np.arange(sp_weight_min, sp_weight_max + sp_weight_step, sp_weight_step)
    tbond_weight_range = np.arange(tbond_weight_min, tbond_weight_max + tbond_weight_step, tbond_weight_step)

    # Run optimization
    if st.button('Run Optimization'):
        best_25th_percentile, best_sp_weight, best_tbond_weight = optimize_portfolio(df, num_years, num_simulations, start_capital, use_inflation_adjusted, interest_basis, sp_weight_range, tbond_weight_range)
        st.write(f"Best 25th Percentile: {best_25th_percentile}%")
        st.write(f"Optimal S&P 500 Weight: {best_sp_weight}")
        st.write(f"Optimal T.Bond Weight: {best_tbond_weight}")
else:
    # Input widgets for S&P and T.Bond weights
    sp_weight = st.number_input('S&P 500 Weight', min_value=0.0, max_value=5.0, value=0.7)
    tbond_weight = st.number_input('T.Bond Weight', min_value=0.0, max_value=5.0, value=0.3)

    # Run the simulation when the button is clicked
    if st.button('Run Simulation'):
        stats_df, final_values_percent, ending_capitals = simulate_portfolio_returns(df, num_years, num_simulations, start_capital, sp_weight, tbond_weight, use_inflation_adjusted, interest_basis)

        # Display the results table
        st.write("Simulation Results:")
        st.dataframe(stats_df)

        # Plot histogram of the final compounded return values in percentages
        fig, ax = plt.subplots()
        ax.hist(final_values_percent, bins=200, edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of Compounded Portfolio Returns ({num_simulations} simulations)')
        ax.set_xlabel('Compounded Return (%)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Plot histogram of the ending capitals
        fig, ax = plt.subplots()
        ax.hist(ending_capitals, bins=200, edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of Ending Capitals ({num_simulations} simulations)')
        ax.set_xlabel('Ending Capital')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

