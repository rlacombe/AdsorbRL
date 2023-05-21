import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm 

def load_transitions(filename='SARS.pkl'):
    """
    Loads the SARS transitions dataset from a pickle file.

    Args:
        filename (str): The name or path of the pickle file containing the SARS transitions dataset.
                        Default is 'SARS.pkl' in the current directory.

    Returns:
        list: A list of transitions, where each transition is a NumPy array containing
              the current state, action, reward, and next state.
    """

    pickleFile = open('SARS.pkl', 'rb')
    SARS = pickle.load(pickleFile)
    pickleFile.close() 
    
    return SARS


def onehot_to_str(material_vector, elements_list):

    material_list = material_vector.tolist()
    material_string = ""

    for i, onehot in enumerate(material_list):
        if onehot == 1: material_string += elements_list[i]

    return material_string


def load_graph(dataset, elements_list):
    """
    Creates a NetworkX directed graph representing state transitions from the given dataset.

    Args:
        dataset (list): A list of transitions, where each transition is a tuple containing the
                        current state, action, reward, and next state.
        elements_list (list): A list of element symbols corresponding to the one-hot encoded
                              representation of the states in the dataset.

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the state transitions.
    """

    G = nx.DiGraph()
    progress_bar = tqdm(total=len(dataset), desc="Creating graph")
    
    for transition in dataset:
        
        # Retrieve allowed transition
        s, a, r, s_prime = transition

        # Transform into strings for node representation
        s, s_prime = onehot_to_str(s, elements_list), onehot_to_str(s_prime, elements_list)
    
        # Add the source state s as a node in the graph if it doesn't exist yet
        if s not in G:
            G.add_node(s)
    
        # Add the target state s' as a node in the graph if it doesn't exist yet
        if s_prime not in G:
            G.add_node(s_prime)
    
        # Add an edge from s to s' with the action a and reward r
        G.add_edge(s, s_prime, reward=r)

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()
    return G

elements_list = [
    "Ag", "Al", "As", "Au", "B", "Bi", "C", "Ca", "Cd", "Cl", "Co", "Cr", "Cs",
    "Cu", "Fe", "Ga", "Ge", "H", "Hf", "Hg", "In", "Ir", "K", "Mn", "Mo", "N",
    "Na", "Nb", "Ni", "Os", "P", "Pb", "Pd", "Pt", "Rb", "Re", "Rh", "Ru", "S",
    "Sb", "Sc", "Se", "Si", "Sn", "Sr", "Ta", "Tc", "Te", "Ti", "Tl", "V", "W",
    "Y", "Zn", "Zr"
]

print("Loading pickle file...\n")
SARS = load_transitions('SARS.pkl')

print("Creating graph...\n")
G = load_graph(SARS, elements_list)

print ("\nEccentricity of example elements:")
ecc = nx.eccentricity(G, ["CSi", "Ag", "C", "Si", "AgZr"])
print(f"Eccentricities: {ecc}\n")

print ("Neighbors of SiC:")
nei_1 = list(nx.neighbors(G, "CSi"))
print(f"# of neighbors: {len(nei_1)}")
nei_2 = []
for n in nei_1: nei_2 = list(set(nei_2).union(set(list(nx.neighbors(G, n)))))
nei_2 = list(set(nei_2) - set(nei_1))
print(f"# of 2nd-degree neighbors: {len(nei_2)}")
nei_3 = []
for n in nei_2: nei_3 = list(set(nei_3).union(list(nx.neighbors(G, n))))
nei_3 = list(set(nei_3) - set(nei_1).union(set(nei_2)))
print(f"# of 3rd-degree neighbors: {len(nei_3)}\n")

print("Computing diameter (max eccentricity)...")
diam = nx.diameter(G)
print(f"Diameter: {diam}\n")

print("Computing radius (min eccentricity)...")
rad = nx.radius(G)
print(f"Radius: {rad}\n")

print("Computing average eccentricity...")
avg_ecc = 0
for n in G.nodes: avg_ecc += nx.eccentricity(G, n)
avg_ecc /= len(G.nodes)
print(f"Average eccentricity: {avg_ecc}")