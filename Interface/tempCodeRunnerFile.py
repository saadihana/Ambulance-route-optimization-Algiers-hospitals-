
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/you', methods=['POST'])
def you():
    address = request.form['address']
    speciality = request.form['speciality']
    search_type = request.form['searchType']
    
    # Fetch patient coordinates
    patient_coordinates = get_patient_coordinates(address)
    if not patient_coordinates:
        return "Failed to fetch patient coordinates.", 400
    
    nearest_route_coordinates__for_patient = find_closest_coordinates(patient_coordinates, 'transition_model.csv')
    if not nearest_route_coordinates__for_patient:
        return "Failed to find the nearest route for the hospital.", 400

    # Find the nearest hospital's coordinates
    nearest_hospital_coordinates = HospitalProblem(None, None, None, None).find_nearest_hospital_with_service(
        patient_coordinates[0], patient_coordinates[1], speciality, 'MY_CSV.csv'
    )
    if not nearest_hospital_coordinates:
        return "No hospitals found for the specified speciality.", 400

    # Find the nearest route for the hospital
    nearest_route_coordinates__for_hospital = find_closest_coordinates(nearest_hospital_coordinates, 'transition_model.csv')
    if not nearest_route_coordinates__for_hospital:
        return "Failed to find the nearest route for the hospital.", 400

    # Create an instance of the HospitalProblem class
    hospital_problem = HospitalProblem(nearest_route_coordinates__for_patient, nearest_route_coordinates__for_hospital, 'transition_model.csv', 100)
    
        # Perform the search
    if search_type == 'hillClimbing':
        result_node = hospital_problem.graph_search(hospital_problem, 'Hill Climbing')
    elif search_type == 'bfs':
        result_node = hospital_problem.graph_search(hospital_problem, 'BFS')
    elif search_type == 'aStar':
        result_node = hospital_problem.graph_search(hospital_problem, 'A*')
    elif search_type == 'ucs':
        result_node = hospital_problem.graph_search(hospital_problem, 'UCS')
    else:
        return "Invalid search type selected.", 400

    if result_node:
        path = []
        while result_node:
            path.append(result_node.state)
            result_node = result_node.parent
        path.reverse()  

        visualize_path_on_map(path)  # Call the function to visualize the path
        return redirect(url_for('map'))
    else:
        return "No solution found.", 400
    

@app.route('/map')
def map():
    return render_template('path_map.html')

if __name__ == '__main__':
    app.run(debug=True)