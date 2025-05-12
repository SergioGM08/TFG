#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''

Example: Monte Carlo Optimization using Simulated Annealing

We optimize a complex non-convex function using Simulated Annealing (SA), 
a Monte Carlo-based method. 

The algorithm explores the solution space randomly, occasionally accepting 
worse solutions to avoid local minima.

Steps:
    
1.	Define a non-convex function to optimize.
2.	Implement the Simulated Annealing (SA) Algorithm.
3.	Visualize the optimization process.

How it Works:
    
•	Step 1: Define a non-convex function.
•	Step 2: Start from an initial guess and explore new points randomly.
•	Step 3: Accept new points if they improve the solution; occasionally accept worse points to escape local minima.
•	Step 4: Gradually reduce the temperature to refine the search.

This method is useful for global optimization in cases where traditional 
gradient-based methods fail.

'''


import numpy as np
import matplotlib.pyplot as plt

# Define an objective function with multiple local minima
def objective_function(x):
    return np.sin(3*x) + 0.5*np.cos(5*x) - 0.1*x**2

# Simulated Annealing algorithm
def simulated_annealing(objective, x_start, temp_start, temp_end, cooling_rate, max_iter):
    x_current = x_start
    best_x = x_current
    best_f = objective(x_current)
    temp = temp_start
    
    history = []  # Store history for visualization
    
    for i in range(max_iter):
        x_new = x_current + np.random.uniform(-0.5, 0.5)  # Random perturbation
        f_new = objective(x_new)
        
        # Accept new solution if it's better or probabilistically accept worse solutions
        if f_new > best_f or np.exp((f_new - best_f) / temp) > np.random.rand():
            x_current = x_new
            best_x, best_f = x_new, f_new
        
        temp *= cooling_rate  # Reduce temperature
        if temp < temp_end:
            break
        
        history.append((x_current, best_f))
    
    return best_x, best_f, history

# Run optimization
x_opt, f_opt, history = simulated_annealing(objective_function, x_start=0, temp_start=10, temp_end=0.01, cooling_rate=0.98, max_iter=500)

# Visualization
x_vals = np.linspace(-5, 5, 1000)
y_vals = objective_function(x_vals)
history_x, history_y = zip(*history)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label='Objective Function')
plt.scatter(history_x, history_y, color='red', alpha=0.5, label='Optimization Path')
plt.scatter([x_opt], [f_opt], color='blue', s=100, label='Optimal Solution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Monte Carlo Optimization using Simulated Annealing')
plt.legend()
plt.show()

# Print result
print(f"Optimal x: {x_opt:.4f}, Optimal f(x): {f_opt:.4f}")
