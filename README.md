# Ambulance path optimization
## Overview

This project presents an system designed to optimize ambulance routing and patient transfer logistics in Algiers using AI algorithms. The goal is to reduce emergency response times and ensure effective distribution of patients across healthcare facilities while considering medical urgency and hospital capacities.

The solution leverages graph-based modeling of the city’s road network, pathfinding algorithms, and constraint satisfaction techniques to dispatch ambulances efficiently. Real-time elements such as traffic data and hospital capacity are incorporated to ensure realistic and adaptive decision-making.

---

## Table of Contents
1. Problem Statement  
2. Solution Architecture  
3. Implementation Details  
4. Algorithms Used  
5. Results & Analysis

---

## Problem Statement

Emergency medical services in urban areas like Algiers face challenges in efficiently routing ambulances and transferring patients between hospitals. These challenges include:

- Minimizing ambulance response and travel times  
- Assigning patients to appropriate hospitals with available capacity  
- Handling two patient priority levels:  
  - Red Code: Critical, urgent cases (1 per ambulance)  
  - Green Code: Less urgent cases (up to 3 per ambulance)  
- Respecting constraints like service availability and ambulance allocation  

The core task is to match patients with hospitals through optimized, constraint-aware routes.

---

## Solution Architecture

### 1. Road Network Graph
- Nodes = road intersections
- Edges = road segments with distance weights
- Data from OpenStreetMap (via Python script)

### 2. Hospital Dataset
- Manually collected from hospital websites
- Includes:
  - Geographic coordinates
  - Service types (e.g., cardiology)
  - Bed capacity by service

### 3. Optimization Engine
- Combines pathfinding and constraint satisfaction
- Simulates traffic and velocity variations
- Matches ambulances and patients to hospitals based on needs and availability

---

## Implementation Details

### Key Classes

- **HospitalProblem**
  - `successors()`: Generates next states
  - `graph_search(strategy)`: Applies BFS, UCS, A*, etc.
  - `hill_climbing_search()`: Optimizes local route planning
  - `find_nearest_hospital_with_service()`: Matches patients to valid hospitals

- **CSP**
  - Handles ambulance routing constraints
  - `constructive_heuristic()`: Generates valid dispatch strategies under limits (e.g., red/green code)

- **Graph**
  - `add_edge()`, `get_neighbors()`: Models and queries the road network

### Example Constraint Logic

```python
if patient.priority == 'RED':
    ambulance.capacity = 1
    must_transfer = True
elif patient.priority == 'GREEN':
    ambulance.capacity = 3
    can_hold = True
```
##Algorithms Used
| Algorithm     | Purpose             | Highlights                                      |
|---------------|---------------------|-------------------------------------------------|
| A*            | Optimal pathfinding | Uses heuristics for smart, efficient routing    |
| Hill Climbing | Local optimization  | Greedy and fast; may use IDA* to improve        |
| IDA*          | Fallback optimizer  | Combines DFS and A* for full coverage           |
| BFS / UCS     | Baselines           | BFS is exhaustive; UCS ensures lowest cost      |

