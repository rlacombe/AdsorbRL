import pickle
import numpy
from tqdm import tqdm 
import networkx as nx
import matplotlib.pyplot as plt

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

    G = nx.Graph()
    progress_bar = tqdm(total=len(dataset), desc="Creating graph")
    
    for transition in dataset:
        
        # Retrieve allowed transition
        s, _, r, s_prime = transition

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


def draw_graph(G):
    print("Computing graph layout...\n")
    pos=nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    print("Plotting graph...\n")
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color='lightblue', node_size=5, font_size=12, font_weight='bold')
    plt.show()


def draw_ego_graph(G, node):
    ego_g = nx.ego_graph(G, node)
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()

    # Draw graph
    pos = nx.spring_layout(ego_g, seed=1)  # Seed layout for reproducibility
    nx.draw(ego_g, pos=pos, node_color="lightblue", edge_color="lightgray", font_color="darkblue", font_weight="normal", node_size=50, with_labels=True, ax=ax)

    # Draw ego as large and red
    options = {"node_size": 500, "node_color": "lightblue"}
    nx.draw_networkx_nodes(ego_g, pos=pos, nodelist=[node], **options)
    plt.show()


def draw_2hop_graph(G, central_node):

    # Make a graph of the single central node
    ego_graph_0hop = nx.ego_graph(G, central_node, radius=0)

    # Compute the 1-hop ego graph (neighborhood) around the central node
    ego_graph_1hop = nx.ego_graph(G, central_node, radius=1)

    # Compute the 2-hop ego graph (neighborhood) around the central node
    ego_graph_2hop = nx.ego_graph(G, central_node, radius=2)
    
    # Setup the graph
    _, ax = plt.subplots(figsize=(16, 8))
    pos = nx.spring_layout(ego_graph_2hop, seed=1)  # Seed layout for reproducibility

    # Draw the graphs
    nx.draw(ego_graph_2hop, pos=pos, node_color="cornflowerblue", edge_color="lightgray", font_color="navy",\
            font_weight="normal", font_size=7, node_size=30, with_labels=True, width=0.05, ax=ax)
    nx.draw(ego_graph_1hop, pos=pos, node_color="palegreen", edge_color="lightgray", font_color="forestgreen",\
            font_weight="normal", font_size=7, node_size=30, with_labels=True, width=0.05, ax=ax)
    nx.draw(ego_graph_0hop, pos=pos, node_color="coral", edge_color="lightgray", font_color="orangered",\
            font_weight="normal", font_size=7, node_size=250, with_labels=True, width=0.05, ax=ax)
    plt.show()


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

draw_2hop_graph(G, "CSi")

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