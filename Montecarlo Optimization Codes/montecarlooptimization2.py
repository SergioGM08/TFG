#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''

Real-World Example: Monte Carlo Portfolio Optimization

One practical use of Monte Carlo optimization is in portfolio optimization. 
Investors aim to allocate assets to maximize expected return while minimizing 
risk (variance). 

Monte Carlo methods help find an optimal portfolio allocation by randomly 
sampling different portfolio weight combinations and selecting the best one.
 
Steps:
    
1.	Generate random portfolios with different asset allocations.
2.	Compute portfolio return and risk (standard deviation).
3.	Find the best allocation maximizing the Sharpe ratio (return per unit risk).
4.	Visualize the Efficient Frontier.

How It Works:
    
1.	Generate Random Portfolios: Each portfolio has randomly assigned weights 
    to different assets.
2.	Compute Returns & Risk: Using asset statistics, we compute portfolio return 
    and standard deviation.
3.	Sharpe Ratio Maximization: The Sharpe ratio measures return per unit of risk; 
    we select the portfolio with the highest value.
4.	Visualization: The Efficient Frontier is plotted, and the optimal portfolio 
    is highlighted.

This method helps investors diversify investments and find optimal risk-adjusted 
returns.

'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulated asset returns (mean annual return, standard deviation)
assets = {
    "Stock A": (0.12, 0.25),
    "Stock B": (0.08, 0.15),
    "Stock C": (0.10, 0.20),
    "Bond D": (0.05, 0.10)
}

# Number of random portfolios to generate
num_portfolios = 10000
risk_free_rate = 0.02  # 2% risk-free rate

# Extract data
asset_names = list(assets.keys())
mean_returns = np.array([assets[a][0] for a in asset_names])
std_devs = np.array([assets[a][1] for a in asset_names])
cov_matrix = np.diag(std_devs**2)  # Simplified, assuming no correlation

# Monte Carlo Simulation
portfolio_returns = []
portfolio_risks = []
portfolio_weights = []
sharpe_ratios = []

for _ in range(num_portfolios):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)  # Normalize to sum to 1
    
    # Calculate portfolio performance
    port_return = np.sum(weights * mean_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_risk
    
    # Store results
    portfolio_returns.append(port_return)
    portfolio_risks.append(port_risk)
    portfolio_weights.append(weights)
    sharpe_ratios.append(sharpe_ratio)

# Convert to DataFrame
data = pd.DataFrame({
    "Return": portfolio_returns,
    "Risk": portfolio_risks,
    "Sharpe Ratio": sharpe_ratios
})

# Find optimal portfolio (max Sharpe ratio)
optimal_idx = np.argmax(sharpe_ratios)
optimal_portfolio = portfolio_weights[optimal_idx]
optimal_return = portfolio_returns[optimal_idx]
optimal_risk = portfolio_risks[optimal_idx]

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_risk, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Portfolio Optimization')
plt.legend()
plt.show()

# Print results
print(f"Optimal Portfolio Weights:")
for i, asset in enumerate(asset_names):
    print(f"{asset}: {optimal_portfolio[i]:.2%}")
print(f"Expected Return: {optimal_return:.2%}, Risk: {optimal_risk:.2%}")
