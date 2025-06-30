
# THIS FILE AIMS TO CONSTRUCT THE TRANSITION MODEL FROM route_connections.csv

import csv

def read_routes_from_csv(csv_file):
    routes = []
    current_route = []
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Node ID']:  # New route starts
                if current_route:  # Save previous route
                    routes.append(current_route)
                    current_route = []
                current_route.append({
                    'Coordinates': (float(row['Latitude']), float(row['Longitude'])),
                    'Connected Routes': []
                })
            else:  # Connected route
                current_route[-1]['Connected Routes'].append({
                    'Coordinates': (float(row['Neighbor Latitude']), float(row['Neighbor Longitude'])),
                    'Distance': float(row['Distance (meters)'])
                })
        if current_route:  # Save the last route
            routes.append(current_route)
    return routes

def write_transition_model_to_csv(routes, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Route Coordinates', 'Connected Route Coordinates', 'Distance'])
        for route in routes:
            route_coordinates = route[0]['Coordinates']
            for connection in route[0]['Connected Routes']:
                connected_route_coordinates = connection['Coordinates']
                distance = connection['Distance']
                writer.writerow([route_coordinates, connected_route_coordinates, distance])

# Usage example:
routes = read_routes_from_csv('route_connections.csv')
write_transition_model_to_csv(routes, 'transition_model.csv')


