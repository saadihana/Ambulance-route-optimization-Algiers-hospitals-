import random
import math
import folium
                    
# Hospitals
hospitals = {
    "CHU Mustapha Pacha d'Alger": {"location": (36.76202550605904, 3.056338988357211), "capacity": 1500, "services": ["neonatalogie", "parodontologie", "reeducation fonctionnelle", "ophtalmologie", "hepatologie", "diabetologie", "chirurgie thoracique", "gastroenterologie", "clinique chirurgicale e B e", "dermatologie et venereologie", "immunologie", "parasitologie et mycologie", "medecine legale", "clinique chirurgicale e A e", "service neurologie", "pathologie et chirurgie buccale"]},
    "CHU Lamine Debaghine de Bab El Oued (ex - Maillot)": {"location": (36.79493412764575, 3.0495627423292495), "capacity": 1200, "services": ["medecine nucleaire", "pediatrie", "orthopedie traumatologie", "urgences medico-chirurgicales", "endocrinologie", "diabetologie", "anesthesie reanimation", "pneumologie", "reanimation medicale", "anatomie pathologique", "ophtalmologie", "neurologie", "hemobiologie et centre de transfusion sanguine", "dermatologie pediatrique", "gastro-enterologie", "neurochirurgie", "chirirgie generale", "medecine du travail", "medecine legale", "medecine preventive", "gynecologie obstetrique", "urologie", "d'oto-rhyno-laryngologie", "medecine interne", "rhumatologie", "psychiatrie"]},
    "CHU Nafissa Hamoud de Hussein Dey (ex - Parnet)": {"location": (36.736675274254004, 3.1057623595183768), "capacity": 1080, "services": ["Radiologue", "Physiologue", "Pediatrie", "Parasitologue", "Ophtalmologue", "Nephrologue", "Medecine Legale", "Histologue", "Gyneco-Obstetrique", "Epidemiologue", "Chirurgie"]},
    "CHU Issad Hassani de Beni Messous": {"location": (36.77840266694043, 2.9821742170495233), "capacity": 880, "services": ["Pneumo-Allergologue", "Rhumatologue", "Reanimation", "Radiologue", "Pneumologue", "Pneumo-Phtisiologue", "Pediatrie", "ORL", "Ophtalmologue", "Nephrologue", "Microbiologue", "Medecine du Travail", "Medecine Interne", "Hematologue", "Gyneco-Obstetrique", "Epidemiologue", "Chirurgie Generale", "Chirurgie Pediatrique", "Cardiologue", "Anatomie Pathologique"]},
    "CHU Djillali Bounaema de Douera": {"location": (36.66682836548595, 2.9427430676355346), "capacity": 790, "services": ["Chirurgie des Breles", "Chirurgie Orthopedique"]},
    "EHS en Oncologie Pierre et Marie Curie (CPMC)": {"location": (36.76080368488759, 3.0542441456052276), "capacity": 230, "services": ["Pathologie Cancereuse"]},
    "Clinique chirurgicale Debussy (annexe du CPMC)": {"location": (36.76530090666471, 3.047691801847626), "capacity": 190, "services": ["Consultations chirurgicales", "evaluations preoperatoires", "Interventions chirurgicales", "Soins postoperatoires", "Services de readaptation"]},
    "Etablissement Public de Sante de Proximite de Douera": {"location": (36.67091035214355, 2.9525991629339257), "capacity": 150, "services": ["Orthopedie -A-", "Orthopedie -B-", "Medecine physique et reeducation fonctionnelle", "Rhumatologie", "Chirurgie plastique", "Anatomie pathologique", "Chirurgie generale", "Medecine interne", "Cardiologie", "Chirurgie maxillo-faciale", "pediatrie", "Radiologie", "Gynecologie", "medecine legale"]},
    "EHS des brulus Pierre et Claudine Chaulet": {"location": (36.77267620017187, 3.0568079460286457), "capacity": 320, "services": ["Chirurgie Plastique Reconstructrice", "Chirurgie des Breles"]},
    "EHS Maouche Mohand Amokrane (ex-CNMS) Specialite : Cardiologie": {"location": (36.767863332880644, 3.001459613305552), "capacity": 180, "services": ["Anesthesie - Reanimation", "Cardiologue", "Chirurgie Cardiaque", "Chirurgie Vasculaire", "Medecine du Sport", "Physiologue"]},
    "EHS Mohamed Abderrahmani en chirurgie cardiovasculaire": {"location": (36.73661901830541, 3.053271387701236), "capacity": 50, "services": ["Consultation medicale", "Traitement", "Soins palliatifs", "Examens diagnostiques", "Readaptation"]},
    "EHS Mere - enfant Hassen Badi - (ex-Belfort)": {"location": (36.72199644439113, 3.1477551877739764), "capacity": 80, "services": ["Gyneco-Obstetrique", "Pediatrie"]},
    "EHS Hopital des Urgences Medico-Chirurgicales Salim ZEMIRLI EPA": {"location": (36.709400522721715, 3.120935518494314), "capacity": 240, "services": ["Urgences Medicaux Chirurgicales", "Orthopedie Neuro-Chirurgue", "Medecine Legale Medecine", "Interne Chirurgie", "Generale Anesthesie - Reanimation"]},
    "EHS - Hopital Psychiatrique Drid Hocine": {"location": (36.73917036611551, 3.0853057152018963), "capacity": 280, "services": ["Psychiatrie"]},
    "EHS - Hopital Psychiatrique Mahfoud Boucebci": {"location": (36.766447656643855, 2.9552944423292495), "capacity": 120, "services": ["Psychiatrie"]},
    "EHS en MPR (Reeducation fonctionnelle) - Azur Plage": {"location": (36.73576662358241, 2.8453536171115394), "capacity": 220, "services": ["Reeducation et readaptation fonctionnelle"]},
    "EHS Pr Abdelkader Boukhroufa": {"location": (36.75469085275399, 3.001101267622272), "capacity": 80, "services": ["Oncologie medicale", "Oncologie chirurgicale", "Soins de soutien", "Oncologie radiologique"]},
    "EHS/ MPR (Reeducation fonctionnelle)Tixeraene": {"location": (36.72528868631248, 3.0205487539967493), "capacity": 920, "services": ["Reeducation", "readaptation fonctionnelle"]},
    "EHS en Infectiologie El Hadi Flici (ex El Kettar)": {"location": (36.7857207845617, 3.052063709211112), "capacity": 240, "services": ["Urgences Radiologue", "Maladies Infectieuses", "Microbiologue"]},
    "EHS en Neurologie Ali Aet Idir": {"location": (36.787499761598944, 3.05814624232925), "capacity": 220, "services": ["Radiologue", "Neurologue", "Neuro-Chirurgue", "Neurophysiologue"]},
    "EHS Gastrologie Djillali Rahmouni (ex-Clinique les Orangers)": {"location": (36.75513524736968, 3.0419556171891275), "capacity": 120, "services": ["gastrologie"]},
    "EPH Bachir Mentouri": {"location": (36.726269090213314, 3.0870928478721287), "capacity": 220, "services": ["Radiologue", "Medecine Legale", "Gyneco-Obstetrique", "Chirurgie Generale", "Anesthesie - Reanimation", "ORL", "Medecine Interne"]},
    "EPH Djillali Belkhenchir": {"location": (36.77657869677934, 3.0409917295620263), "capacity": 220, "services": ["CHIRURGIE GENERALE", "CHIRURGIE INFANTILE", "PeDIATRIE", "Radiologie Centrale", "Epidemiologie", "MeDECINE INTERNE"]},
    "EPH Belkacemi Tayeb": {"location": (36.70580083800539, 2.8458142729213995), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie Microbiologue", "Medecine Interne", "Gyneco-Obstetrique", "Chirurgie Generale"]},
    "EPH de Rouiba": {"location": (36.73533705438, 3.2858348441768754), "capacity": 140, "services": ["urgences", "chirurgie", "radiologie", "pneumophtisiologie", "oncologie", "medecine interne", "pediatrie"]},
    "EPH - etablissement Public Hospitalier Ibn Ziri": {"location": (36.81641984642462, 2.980246381993367), "capacity": 90, "services": ["Medecine generale", "dermatologie", "endocrinologie", "pediatrie", "Maternite", "radiologie", "cardiologie", "Service des urgences", "imagerie", "Services chirurgicaux."]},
    "Etablissement Public Hospitalier de Ain Taya": {"location": (36.78921789556811, 3.2946943730122507), "capacity": 140, "services": ["Soins hospitaliers", "Services d'urgence"]},
    "L'hopital des grands brules de Zeralda": {"location": (36.69787355685087, 2.8444583828108723), "capacity": 140, "services": ["Anesthesie Reanimation", "Chirurgie Plastique Et Reparatrice Pour Adultes", "Chirurgie Plastique Et Reparatrice Pour Enfants"]},
    "Etablissement Public de Sante de Proximite ZERALDA": {"location": (36.71584669836232, 2.843067695833074), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite REGHAIA": {"location": (36.77959824147116, 3.362387701482595), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite KOUBA (LES ANNASSER)": {"location": (36.734058741339126, 3.0611706454681578), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite CHERAGA ( BOUCHAOUI)": {"location": (36.74408083827578, 2.9133863728726475), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite BOUZAREAH": {"location": (36.79387823442046, 3.0185775091927067), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite BORDJ EL KIFFAN (DERGANA)": {"location": (36.793328369914924, 3.2613066807220674), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite BARAKI": {"location": (36.653700387642125, 3.0905215288353753), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite BAB EL OUED": {"location": (36.78923069967209, 3.049408272792376), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Hospitalier Specialise PSYCHIATRIQUE DRID HOCINE": {"location": (36.74212419997733, 3.0867501918134392), "capacity": 280, "services": ["Psychiatrie"]},
    "Etablissement Public de Sante de Proximite de DRARIA": {"location": (36.73181977455254, 2.9990557109407305), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]},
    "Etablissement Public de Sante de Proximite SIDI MeHAMED (BOUCHENAFA)": {"location": (36.767321075660355, 3.0521513263032674), "capacity": 120, "services": ["Chirurgie dentaire", "Pediatrie", "medcine generale", "Medecine Interne", "Gyneco-Obstetrique", "urgences"]}
}

# Ambulances
ambulances = {
    "a1": {"location": hospitals["CHU Mustapha Pacha d'Alger"]["location"], "idle_time": 0},
    "a2": {"location": hospitals["CHU Lamine Debaghine de Bab El Oued (ex - Maillot)"]["location"], "idle_time": 0},
    "a3": {"location": hospitals["EHS/ MPR (Reeducation fonctionnelle)Tixeraene"]["location"], "idle_time": 0},
}

# Patients
patients = {
    "r1": {"location": (36.76, 3.05), "type": "red", "service_time": 30, "needs": ["Pediatrie"]},
    "r2": {"location": (36.78, 3.04), "type": "red", "service_time": 30, "needs": ["Anesthesie Reanimation"]},
    "r3": {"location": (36.79, 3.03), "type": "red", "service_time": 30, "needs": ["Urgences Medicaux Chirurgicales"]},
    "g8": {"location": (36.70, 2.98), "type": "green", "service_time": 10},
    "g9": {"location": (36.70, 2.98), "type": "green", "service_time": 10},
    "g1": {"location": (36.74, 3.02), "type": "green", "service_time": 10},
    "g2": {"location": (36.73, 3.01), "type": "green", "service_time": 10},
    "g3": {"location": (36.75, 3.03), "type": "green", "service_time": 10},
    "g4": {"location": (36.72, 3.00), "type": "green", "service_time": 10},
    "r4": {"location": (36.71, 2.99), "type": "red", "service_time": 30, "needs": ["Interventions chirurgicales"]},
    "g6": {"location": (36.70, 2.98), "type": "green", "service_time": 10},
    "g7": {"location": (36.69, 2.97), "type": "green", "service_time": 10},
}


# Function to calculate Euclidean distance
def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

# Function to find nearest hospitals or patients
def find_nearest_entities(location, entities, alpha):
    distances = [(entity_id, calculate_distance(location, info["location"])) for entity_id, info in entities.items()]
    distances.sort(key=lambda x: x[1])
    return [entity_id for entity_id, _ in distances[:alpha]]

def constructive_heuristic(ambulances, patients, hospitals, alpha=2):
    solution = {ambulance_id: [] for ambulance_id in ambulances}
    unvisited_patients = set(patients.keys())
    
    while unvisited_patients:
        for ambulance_id in ambulances:
            ambulance = ambulances[ambulance_id]
            hospital_id = find_nearest_entities(ambulance["location"], hospitals, 1)[0]
            route = [hospital_id]
            green_patients_count = 0  # Track the number of green patients
            
            while unvisited_patients:
                nearest_patients = find_nearest_entities(ambulance["location"], {pid: patients[pid] for pid in unvisited_patients}, alpha)
                if not nearest_patients:
                    break
                next_patient_id = random.choice(nearest_patients)
                next_patient = patients[next_patient_id]
                
                if next_patient["type"] == "red":
                    route.append(next_patient_id)
                    unvisited_patients.remove(next_patient_id)
                    suitable_hospitals = [h for h in hospitals if any(service in hospitals[h]["services"] for service in next_patient["needs"])]
                    if suitable_hospitals:
                        hospital_id = random.choice(suitable_hospitals)
                        ambulance["idle_time"] += next_patient["service_time"]
                        route.append(hospital_id)
                    else:
                        hospital_id = find_nearest_entities(next_patient["location"], hospitals, 1)[0]
                        ambulance["idle_time"] += calculate_distance(ambulance["location"], hospitals[hospital_id]["location"])
                        route.append(hospital_id)
                    break
                else:
                    if green_patients_count < 3:  # Check if ambulance can pick up more green patients
                        ambulance["location"] = next_patient["location"]
                        route.append(next_patient_id)
                        unvisited_patients.remove(next_patient_id)
                        green_patients_count += 1
                    else:
                        hospital_id = find_nearest_entities(next_patient["location"], hospitals, 1)[0]
                        ambulance["idle_time"] += calculate_distance(ambulance["location"], hospitals[hospital_id]["location"])
                        route.append(hospital_id)
                        break
            
            solution[ambulance_id].append(route)
            ambulance["location"] = hospitals[hospital_id]["location"]
            ambulance["idle_time"] = sum(patients[pid]["service_time"] for pid in route if pid in patients)

    return solution


# Main function to execute the algorithm
def main():
    # Call the generate_map function to create the map and display ambulance routes
    solution = generate_map(ambulances, patients, hospitals)
    print("Map generated successfully.")
    # Print the solution
    print_solution(solution)

# Function to generate map and display ambulance routes
def generate_map(ambulances, patients, hospitals):
    # Create a map centered around Algiers
    map_center = [36.75, 3.05]
    mymap = folium.Map(location=map_center, zoom_start=12)

    # Add markers for hospitals
    for hospital, info in hospitals.items():
        folium.Marker(location=info["location"], popup=hospital, icon=folium.Icon(color='blue')).add_to(mymap)

    # Add markers for patients
    for patient, info in patients.items():
        folium.Marker(location=info["location"], popup=patient, icon=folium.Icon(color='red' if info["type"] == "red" else 'green')).add_to(mymap)

    # Call the constructive_heuristic function to get the ambulance routes
    solution = constructive_heuristic(ambulances, patients, hospitals)

    # Add polylines for ambulance routes
    for ambulance, routes in solution.items():
        for route in routes:
            coordinates = []
            for point in route:
                if point in hospitals:
                    coordinates.append(hospitals[point]["location"])
                elif point in patients:
                    coordinates.append(patients[point]["location"])
            folium.PolyLine(locations=coordinates, color='blue').add_to(mymap)

    # Save the map to an HTML file
    mymap.save("ambulance_routes.html")
    print("Map saved as ambulance_routes.html.")
    
    return solution

# Function to print the solution
def print_solution(solution):
    for ambulance_id, routes in solution.items():
        print(f"Ambulance {ambulance_id}:")
        for i, route in enumerate(routes):
            route_str = " -> ".join(route)
            print(f"    Route {i + 1}: {route_str}")

# Conditional check to execute the main function
if __name__ == "__main__":
    main()  # Call the main function
