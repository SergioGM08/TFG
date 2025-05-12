#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''

Below is a Python implementation that combines Monte Carlo Simulation 
and Genetic Algorithms (GA) to solve the Vehicle Routing Problem (VRP).
 
Problem Statement:
    
We have:
    
•	A fleet of vehicles with a limited capacity.
•	A set of customer locations with demand.
•	A central depot where vehicles start and return.

The goal is to:
    
•	Find the optimal routes to minimize the total distance traveled while 
    fulfilling all customer demands.
    
To solve this, we:
    
1.	Use Monte Carlo Simulation to generate random initial routes.
2.	Apply Genetic Algorithm (GA) to evolve the best routes over multiple 
    generations.
 
Python Code Implementation

We use:
    
•	numpy for numerical operations.
•	matplotlib for visualization.
•	deap for the Genetic Algorithm.
•	scipy.spatial.distance for distance calculations.

'''


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from deap import base, creator, tools, algorithms

# Step 1: Generate Random Customers and Depot
num_customers = 15
num_vehicles = 3
vehicle_capacity = 50
customer_demands = np.random.randint(5, 15, size=num_customers)

# Randomly generate customer locations
np.random.seed(42)
customer_locations = np.random.rand(num_customers, 2) * 100  # 100x100 grid
depot_location = np.array([[50, 50]])  # Central depot
all_locations = np.vstack([depot_location, customer_locations])

# Compute distance matrix
distance_mat = distance_matrix(all_locations, all_locations)

# Step 2: Monte Carlo Simulation to Generate Initial Routes
def generate_initial_routes():
    """Generate random feasible routes"""
    customers = list(range(1, num_customers + 1))  # Customer indices (excluding depot)
    random.shuffle(customers)
    routes = []
    
    while customers:
        route = []
        capacity = 0
        while customers and capacity + customer_demands[customers[0] - 1] <= vehicle_capacity:
            customer = customers.pop(0)
            route.append(customer)
            capacity += customer_demands[customer - 1]
        routes.append(route)
    
    return routes

# Step 3: Define Cost Function
def route_cost(routes):
    """Compute total travel distance for a set of routes"""
    total_distance = 0
    for route in routes:
        if route:
            path = [0] + route + [0]  # Start and end at depot
            for i in range(len(path) - 1):
                total_distance += distance_mat[path[i], path[i + 1]]
    return total_distance

# Step 4: Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_initial_routes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    """Evaluate an individual by calculating its route cost"""
    return route_cost(individual),

def crossover(parent1, parent2):
    """Perform crossover between two parents to produce two children"""
    child1, child2 = [], []
    cut1, cut2 = sorted(random.sample(range(1, num_customers), 2))
    
    # Create offspring by swapping segments
    child1_part = sum(parent1, [])[cut1:cut2]
    child2_part = sum(parent2, [])[cut1:cut2]
    
    def fill_child(parent, child_part):
        """Fill child with crossover part and preserve order"""
        child = [c for c in sum(parent, []) if c not in child_part]
        new_routes = []
        while child:
            route = []
            capacity = 0
            while child and capacity + customer_demands[child[0] - 1] <= vehicle_capacity:
                route.append(child.pop(0))
                capacity += customer_demands[route[-1] - 1]
            new_routes.append(route)
        return new_routes
    
    child1 = fill_child(parent1, child1_part)
    child2 = fill_child(parent2, child2_part)
    
    return creator.Individual(child1), creator.Individual(child2)

def mutate(individual):
    """Mutate an individual by swapping two customers within a route"""
    if len(individual) > 1:
        r1, r2 = random.sample(range(len(individual)), 2)
        if individual[r1] and individual[r2]:
            c1, c2 = random.choice(individual[r1]), random.choice(individual[r2])
            individual[r1].remove(c1)
            individual[r2].remove(c2)
            individual[r1].append(c2)
            individual[r2].append(c1)
    return individual,

toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Step 5: Run Genetic Algorithm
population = toolbox.population(n=100)
NGEN = 50
CXPB = 0.7
MUTPB = 0.2

for gen in range(NGEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    population[:] = offspring

# Step 6: Get Best Solution
best_solution = tools.selBest(population, 1)[0]
best_cost = route_cost(best_solution)

# Step 7: Visualization
plt.figure(figsize=(10, 6))
for route in best_solution:
    route_path = np.array([depot_location[0]] + [customer_locations[i - 1] for i in route] + [depot_location[0]])
    plt.plot(route_path[:, 0], route_path[:, 1], marker='o')

plt.scatter(depot_location[0, 0], depot_location[0, 1], color='red', s=200, label="Depot")
plt.scatter(customer_locations[:, 0], customer_locations[:, 1], color='blue', s=100, label="Customers")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title(f"Optimized VRP Solution (Total Distance: {best_cost:.2f})")
plt.legend()
plt.grid()
plt.show()

# Step 8: Print Results
print("\nOptimal Vehicle Routes:")
for i, route in enumerate(best_solution):
    print(f"Vehicle {i + 1}: Depot -> {' -> '.join(map(str, route))} -> Depot")
print(f"\nTotal Optimized Distance: {best_cost:.2f}")
