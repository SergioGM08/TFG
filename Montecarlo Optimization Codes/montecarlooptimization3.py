#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''

Advanced Monte Carlo Optimization: Vehicle Routing Problem (VRP)

A more complex real-world application of Monte Carlo optimization is solving 
the Vehicle Routing Problem (VRP). This problem arises in logistics and 
transportation when multiple vehicles need to deliver goods to different 
locations while minimizing cost, distance, or time.
 
Problem Statement

•	A fleet of vehicles must deliver packages to multiple customers.
•	Each vehicle has a limited capacity.
•	The goal is to minimize the total travel distance while ensuring all 
    customers receive their deliveries.

Monte Carlo Approach

1.	Generate random feasible routes for vehicles.
2.	Compute total travel distance for each route.
3.	Use Monte Carlo simulation to refine routes and find the best one.
 
Python Implementation

This implementation:
    
•	Generates a random set of delivery locations.
•	Uses Monte Carlo sampling to simulate different routing solutions.
•	Selects the best route based on minimum total travel distance.

How This Works

1.	Generate random delivery points within a 100×100 grid.
2.	Compute distances between all locations (Euclidean distance).
3.	Randomly shuffle delivery assignments and simulate vehicle routes.
4.	Evaluate and refine routes using Monte Carlo sampling.
5.	Select the best route with the shortest total distance.
 
Real-World Applications

•	Logistics companies (e.g., Amazon, UPS) use VRP optimization to minimize 
    delivery costs.
•	Food delivery services (e.g., UberEats, DoorDash) optimize routes to 
    improve delivery time.
•	Waste collection & public transport use similar routing techniques.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Number of delivery locations (excluding depot)
n_customers = 10
n_vehicles = 3
vehicle_capacity = 4

# Generate random delivery locations (including depot at (0,0))
np.random.seed(42)
locations = np.random.rand(n_customers + 1, 2) * 100  # Scaled to a 100x100 grid

def total_route_distance(routes, distance_matrix):
    """Calculate the total distance for a given set of vehicle routes."""
    total_distance = 0
    for route in routes:
        if route:  # If vehicle is assigned deliveries
            total_distance += distance_matrix[0, route[0]]  # Depot to first customer
            for i in range(len(route) - 1):
                total_distance += distance_matrix[route[i], route[i + 1]]
            total_distance += distance_matrix[route[-1], 0]  # Last customer back to depot
    return total_distance

# Compute distance matrix (Euclidean distance between locations)
distance_matrix = cdist(locations, locations, metric='euclidean')

# Monte Carlo Simulation Parameters
num_simulations = 10000
best_routes = None
best_distance = float('inf')

# Monte Carlo Optimization Loop
for _ in range(num_simulations):
    np.random.shuffle(locations[1:])  # Shuffle delivery points randomly
    
    # Assign deliveries to vehicles (without exceeding capacity)
    routes = [list(range(i, min(i + vehicle_capacity, n_customers))) for i in range(1, n_customers + 1, vehicle_capacity)]
    
    # Compute distance
    total_distance = total_route_distance(routes, distance_matrix)
    
    # Check if it's the best solution found
    if total_distance < best_distance:
        best_distance = total_distance
        best_routes = routes

# Visualization
plt.figure(figsize=(8, 8))
plt.scatter(locations[:, 0], locations[:, 1], c='blue', label='Delivery Points')
plt.scatter([locations[0, 0]], [locations[0, 1]], c='red', marker='s', s=200, label='Depot')

# Plot best found routes
colors = ['green', 'purple', 'orange', 'brown', 'pink']
for i, route in enumerate(best_routes):
    route_locs = np.vstack(([locations[0]], locations[route], [locations[0]]))
    plt.plot(route_locs[:, 0], route_locs[:, 1], color=colors[i % len(colors)], label=f'Vehicle {i+1}')

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Optimized Vehicle Routes (Monte Carlo Method)")
plt.legend()
plt.show()

# Print Results
print("Best Routes Found:")
for i, route in enumerate(best_routes):
    print(f"Vehicle {i+1}: Depot -> {route} -> Depot")
print(f"Total Optimized Distance: {best_distance:.2f}")