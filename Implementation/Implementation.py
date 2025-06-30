import csv
import time
import requests
from math import isclose
from math import radians, sin, cos, sqrt, atan2
import heapq
import queue
import random
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import csv
import networkx as nx
import matplotlib.pyplot as plt
import folium


class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, start_node, end_node, weight):
        if start_node not in self.adjacency_list:
            self.adjacency_list[start_node] = []
        self.adjacency_list[start_node].append((end_node, weight))

    def get_neighbors(self, node):
        return self.adjacency_list.get(node, [])
    


def visualize_path_on_map(path):
    # Create a map centered around the first coordinate in the path
    map_object = folium.Map(location=path[0], zoom_start=12)

    # Add markers for the path
    for coord in path:
        folium.Marker(coord).add_to(map_object)

    # Add a line connecting the markers to represent the path
    folium.PolyLine(path, color="blue", weight=2.5, opacity=1).add_to(map_object)

    # Display the map
    map_object.save("path_map.html")
    map_object


# def build_graph_from_csv(csv_file):
#     graph = Graph()
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header row
#         for row in reader:
#             start_coords = eval(row[0])
#             end_coords = eval(row[1])
#             distance = float(row[2])
#             graph.add_edge(start_coords, end_coords, distance)
#             graph.add_edge(end_coords, start_coords, distance)  # Assuming bidirectional connections
#     return graph

class Node:
    def __init__(self, state, parent=None, action=None, cost=0 ):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
            
    def __lt__(self, other):
        return self.cost < other.cost
    def node_cost(self):
        return self.cost

def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the earth in km
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculate the change in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Calculate the distance
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

class HospitalProblem:
    def __init__(self, initial_state, goal_state, state_transition_model, num_segments):
        self.initial_state = initial_state  # actual position of patient
        self.goal_state = goal_state  # closest hospital with conditions
        self.state_transition_model = state_transition_model
        self.num_segments = num_segments
    
    def is_goal_test(self, goal_test):
        return goal_test == self.goal_state

    def successors(self, state):
        successors = {}
        with open(self.state_transition_model, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                start_coords = eval(row[0])  # Convert string representation to tuple
                end_coords = eval(row[1])    # Convert string representation to tuple

                # Check if the current state matches either start or end coordinates
                if state == start_coords:
                    successors[end_coords] = float(row[2])
                elif state == end_coords:
                    successors[start_coords] = float(row[2])
        return successors

    
    def expand_node(self, node, strategy=""):
        successors = []
        children = self.successors(node.state)
        if not children:
            print("No children found for state:", node.state)  # Debugging statement
            return successors  # No children to expand

        for coordinates, distance in children.items():
            if (strategy == 'Hill Climbing'):
                print("Distance from", node.state, "to", coordinates, ":", distance)  # Print distance for debugging
            heuristic_cost = self.heuristic(node.state, coordinates)
            # Add distance to the current cost of the node
            total_cost = node.cost + distance
            if strategy == 'A*':
                total_cost += heuristic_cost  # Add heuristic cost for A* search

            successors.append(Node(coordinates, parent=node, action=coordinates, cost=total_cost))

        # Sort successors based on their costs
        successors.sort(key=lambda x: x.cost)
        return successors


    def ida_star(self, start, goal, neighbors, heuristic):
        def search(path, g, bound):
            current = path[-1]
            f = g + heuristic(current, goal)
            if f > bound:
                return f, None
            if current == goal:
                return f, path
            min_bound = float('inf')
            for neighbor in neighbors(current):
                if neighbor not in path:
                    print("Adding", neighbor, "to the frontier")  # Print frontier update
                    path.append(neighbor)
                    t, result_path = search(path, g + self.heuristic(current, neighbor), bound)
                    if result_path is not None:
                        return t, result_path
                    if t < min_bound:
                        min_bound = t
                    path.pop()
            return min_bound, None

        bound = heuristic(start, goal)
        path = [start]
        while True:
            t, result_path = search(path, 0, bound)
            if result_path is not None:
                return result_path
            if t == float('inf'):
                return None
            bound = t

            
    def hill_climbing(self, initial_node):
        current_node = initial_node
        goal_reached = False
        max_iterations = 5000
        max_sideways_moves = 1000
        iterations = 0
        sideways_moves = 0
        previous_states = set()
        best_solution = None  # Initialize best_solution to None
        while not goal_reached:
            print("Current state:", current_node.state)  # Print current state for debugging
            print("Current cost:", current_node.cost)    # Print current cost for debugging

            if self.is_goal_test(current_node.state):
                goal_reached = True
                best_solution = current_node  # Update best_solution when goal is reached
                break

            neighbors = self.expand_node(current_node)
            if not neighbors:
                print("No more neighbors to explore", current_node.state)  # Debugging statement
                break  # Stuck at local maximum, terminate search

            # Select the neighbor with the smallest cost
            best_neighbor = min(neighbors, key=lambda node: node.cost)
            print("Best neighbor cost:", best_neighbor.cost)
            
            if current_node.state in previous_states:
                print("Stuck at local optimum. Switching to IDA* search.")
                result_path = self.ida_star(current_node.state, self.goal_state, self.successors, self.heuristic)  
                if result_path:
                    print("IDA* found a solution:")
                    goal_reached = True
                    return result_path  # Break from the loop and return the path
                else:
                    break  # If IDA* doesn't find a solution, break the loop

            previous_states.add(current_node.state)

            if best_neighbor.cost >= current_node.cost:
                if sideways_moves < max_sideways_moves:
                    current_node = best_neighbor
                    sideways_moves += 1
                else:
                    print("Too many sideways moves. Restarting search.")
                    current_node = initial_node
                    previous_states.clear()
                    sideways_moves = 0
            else:
                current_node = best_neighbor
                sideways_moves = 0  # Reset sideways moves

            iterations += 1
            if iterations >= max_iterations:
                print("Max iterations reached. Restarting search.")
                current_node = initial_node
                previous_states.clear()
                iterations = 0

        return best_solution  # Return the best solution found by hill climbing




    def calculate_costs(self, node):
            with open('transition_model.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if str(node.state) == row[0]:
                        node.cost = float(row[2])
                        break
    def steepest_ascent(self, current_node, children):
        if not children:
            return None
        best_neighbor = min(children, key=lambda node: node.heuristic_cost)  # Use heuristic cost for comparison
        if best_neighbor.cost >= current_node.cost:
            return None  # Local maximum found
        return best_neighbor
    
    def graph_search(self, problem, strategy):
        if strategy == 'BFS':
            frontier = queue.Queue()
        elif strategy == 'UCS' or strategy == 'A*':
            frontier = queue.PriorityQueue()
        elif strategy == 'Hill Climbing':
            initial_node = Node(self.initial_state)
            return self.hill_climbing(initial_node)  # Call hill_climbing directly
        elif strategy == 'IDA*':
            return self.ida_star(problem)
        else:
            raise ValueError("Invalid strategy")

        initial_node = Node(self.initial_state)
        
        if strategy != 'Hill Climbing':
            frontier.put(initial_node)

        explored = set()
        current_node = None  # Initialize current node outside the loop

        while (strategy != 'Hill Climbing' and not frontier.empty()) or (strategy == 'Hill Climbing'):
            if strategy != 'Hill Climbing':
                current_node = frontier.get()

            if problem.is_goal_test(current_node.state):
                return current_node

            explored.add(current_node.state)

            if strategy == 'Hill Climbing':
                next_node = HospitalProblem.steepest_ascent(current_node, problem.expand_node(current_node.state))
                if next_node is None:
                    return current_node  # Local maximum found
                current_node = next_node  # Move to the best neighbor in hill climbing
            else:
                for child in problem.expand_node(current_node, strategy):  # Pass strategy to expand_node
                    if child.state not in explored:
                        frontier.put(child)
                        print("Adding child to frontier:", child.state)  # Print the child added to the frontier
                        print("Frontier:", [node.state for node in frontier.queue])  # Print the updated frontier
                        print("Explored:", explored)  # Print the current set of explored states
            
        return None






    def find_nearest_hospital_with_service(self, patient_latitude, patient_longitude, required_service, csv_file):
        # Read hospitals from the CSV file
        hospitals = []
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                hospital_name = row[0]
                hospital_latitude = float(row[1])
                hospital_longitude = float(row[2])
                hospital_capacity = int(row[4])
                services = [(row[i], int(row[i + 1]), int(row[i + 2])) for i in range(5, len(row) - 2, 3)]

                # Check if the hospital offers the required service and has available capacity
                for service_name, service_capacity, service_free_capacity in services:
                    if service_name.lower() == required_service.lower() and service_free_capacity > 0 and service_free_capacity < service_capacity:
                        hospitals.append({
                            'name': hospital_name,
                            'latitude': hospital_latitude,
                            'longitude': hospital_longitude,
                            'capacity': hospital_capacity
                        })
                        break  # Stop checking other services if the required service is found

        # Calculate distances between patient and hospitals
        nearest_hospital = None
        nearest_distance = float('inf')
        for hospital in hospitals:
            distance = calculate_distance(patient_latitude, patient_longitude, hospital['latitude'], hospital['longitude'])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_hospital = hospital

        if nearest_hospital:
            return nearest_hospital['latitude'], nearest_hospital['longitude']
        else:
            return None

    def heuristic(self, start_segment, end_segment):
        traffic_data = self.generate_traffic_data()
        max_speed_data = self.generate_max_speeds()
        if isinstance(start_segment, tuple) and isinstance(end_segment, tuple):
            # Extract start and end coordinates from segment names
            start_coordinates = start_segment
            end_coordinates = end_segment

            # Calculate Euclidean distance between start and end coordinates
            euclidean_distance = sqrt((end_coordinates[0] - start_coordinates[0])**2 + (end_coordinates[1] - start_coordinates[1])**2)

            start_speed = max_speed_data.get(start_segment, 120)  # Default to 120 km/h if no speed data available
            end_speed = max_speed_data.get(end_segment, 120)  # Default to 120 km/h if no speed data available

            estimated_time = euclidean_distance / max(start_speed, end_speed)

            start_traffic = traffic_data.get(start_segment, 0)  # Default to 0% traffic congestion if no data available
            end_traffic = traffic_data.get(end_segment, 0)  # Default to 0% traffic congestion if no data available
            traffic_factor = min(start_traffic, end_traffic) / 100  
            adjusted_time = estimated_time * (1 - traffic_factor)

            estimated_distance = adjusted_time * max(start_speed, end_speed)

            return estimated_distance
        else:
            print("Error: Invalid coordinate format. Please provide coordinates as tuples.")
            return None  # Return None in case of error


    def generate_traffic_data(self):
        traffic_data = {}
        for i in range(1, self.num_segments + 1):
            traffic_data[f"Road Segment {i}"] = random.randint(0, 100)
        return traffic_data

    def generate_max_speeds(self, max_speed_range=(30, 120)):
        min_speed, max_speed = max_speed_range
        
        max_speed_data = {}
        for i in range(1, self.num_segments + 1):
            max_speed_data[f"Road Segment {i}"] = random.randint(min_speed, max_speed)  # Random speed within range
        return max_speed_data


def get_patient_coordinates(address):
    geolocator = Nominatim(user_agent="my_geocoder")
    location_data = geolocator.geocode(address)
    if location_data:
        return location_data.latitude, location_data.longitude
    else:
        print("Location not found.")
        return None, None

def find_closest_coordinates(patient_coordinates, csv_file):
    patient_latitude, patient_longitude = patient_coordinates
    closest_distance = float('inf')
    closest_coordinates = None
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            start_coords = tuple(map(float, row[0].strip('()').split(', ')))
            end_coords = tuple(map(float, row[1].strip('()').split(', ')))
            # print("Start coordinates:", start_coords)
            # print("End coordinates:", end_coords)
            start_distance = calculate_distance(patient_latitude, patient_longitude, start_coords[0], start_coords[1])
            end_distance = calculate_distance(patient_latitude, patient_longitude, end_coords[0], end_coords[1])
            if start_distance < closest_distance:
                closest_distance = start_distance
                closest_coordinates = start_coords
            if end_distance < closest_distance:
                closest_distance = end_distance
                closest_coordinates = end_coords
    return closest_coordinates

def print_solution(strategy, result_node):
    if result_node is None:
        print(f"No solution found with strategy {strategy}")
        return
    
    path = []  # Initialize an empty path
    if strategy != 'Hill Climbing':
        while result_node is not None:
            path.append(result_node.state)
            result_node = result_node.parent
        path.reverse()
    else:
        path = result_node  # For hill climbing, the result_node is already the path
    
    print("Path found with strategy", strategy, ":", path)



def find_nearest_route_for_hospital(patient_address, hospital_csv_file, specialty):
    # Create an instance of the HospitalProblem class
    hospital_problem = HospitalProblem(None, None, None, None)

    # Fetch patient coordinates
    patient_coordinates = get_patient_coordinates(patient_address)
    if not patient_coordinates:
        print("Failed to fetch patient coordinates.")
        return None

    # Find the nearest hospital to the patient's location
    nearest_hospital_coordinates = hospital_problem.find_nearest_hospital_with_service(patient_coordinates[0], patient_coordinates[1], specialty, hospital_csv_file)
    if not nearest_hospital_coordinates:
        print("No hospitals found in the provided CSV file.")
        return None

    # Find the nearest route for the hospital
    nearest_route_coordinates = find_closest_coordinates(nearest_hospital_coordinates, 'transition_model.csv')
    if not nearest_route_coordinates:
        print("Failed to find the nearest route for the hospital.")
        return None

    return nearest_route_coordinates


def main():
    patient_address = "CEM Beni Messous El Djadida, Route Celeste, Sidi Youcef, Beni Messous, Daïra Bouzareah, Alger, 16001, Algérie"
    # Patient adress was chosen for demonstrative purposes since in general search takes time and requires advanced hardware 
    hospital_csv_file = 'MY_CSV.csv'
    specialty = "pediatrie"

    # Fetch patient coordinates
    patient_coordinates = get_patient_coordinates(patient_address)
    if not patient_coordinates:
        print("Failed to fetch patient coordinates.")
        return
    else:
        print("Patient coordinates are:")
        print(patient_coordinates)

    # Find the nearest route for the patient
    nearest_route_coordinates_for_patient = find_closest_coordinates(patient_coordinates, 'transition_model.csv')
    if not nearest_route_coordinates_for_patient:
        print("Failed to find the nearest route for the patient.")
        return
    else:
        print("The closest route for the patient is:")
        print(nearest_route_coordinates_for_patient)

    # Find the nearest hospital's coordinates
    nearest_hospital_coordinates = HospitalProblem(None, None, None, None).find_nearest_hospital_with_service(patient_coordinates[0], patient_coordinates[1], specialty, hospital_csv_file)
    if not nearest_hospital_coordinates:
        print("No hospitals found with the required service.")
        return
    else:
        print("The nearest hospital's coordinates are:")
        print(nearest_hospital_coordinates)

    # Find the nearest route for the hospital
    nearest_route_coordinates_for_hospital = find_closest_coordinates(nearest_hospital_coordinates, 'transition_model.csv')
    if not nearest_route_coordinates_for_hospital:
        print("Failed to find the nearest route for the hospital.")
        return
    else:
        print("The closest route for the hospital is:")
        print(nearest_route_coordinates_for_hospital)

    # Create a hospital problem instance
    num_segments = 0
    hospital_problem = HospitalProblem(initial_state=nearest_route_coordinates_for_patient, goal_state=nearest_route_coordinates_for_hospital, state_transition_model="transition_model.csv", num_segments=num_segments)
    # Define search strategies to test
    search_strategies = ['BFS','UCS','A*','Hill Climbing']

    # Initialize current node
    current_node = Node(hospital_problem.initial_state)

    # Test each search strategy
    for strategy in search_strategies:
        print(f"Testing search strategy: {strategy}")
        start_time = time.time()
        # Use graph_search method to find the solution
        if strategy == 'Hill Climbing':
            result_path = hospital_problem.graph_search(hospital_problem, strategy)
        else:
            result_path = hospital_problem.graph_search(hospital_problem, strategy)
        end_time = time.time()
        execution_time = end_time - start_time
        print_solution(strategy, result_path)

        # Visualize the path if a solution is found
        if result_path:
            if strategy == 'Hill Climbing':
                path = result_path
            else:
                path = []
                while result_path:
                    path.append(result_path.state)
                    result_path = result_path.parent
                path.reverse()  # Reverse the path to start from the initial state
            visualize_path_on_map(path)  # Visualize the path on a map

        # Print execution time
        print(f"Execution time for {strategy}: {execution_time} seconds")

        # Write timing data to a CSV file
        with open('timing_data.csv', 'w', newline='') as csvfile:
            fieldnames = ['Strategy', 'Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Strategy': strategy, 'Time': execution_time})

main()
