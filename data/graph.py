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

print("Loading pickle file...")
SARS = load_transitions('SARS.pkl')

print("Creating graph...")
G = load_graph(SARS, elements_list)

print("Computing diameter...")
print(f"Diameter: {nx.diameter(G)}")