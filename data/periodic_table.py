import numpy as np
from tqdm import tqdm 
import networkx as nx
import csv
import re


class Element:
    def __init__(self, atomic_number):
        self.data = {'Z': atomic_number}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def one_hot_encode(self, MAXZ): # Encode an element into a 1-hot state vector
        encoded = np.zeros(MAXZ)
        encoded[self.data['Z']-1] = 1 # NOTE: Z = index + 1 (since Z starts at 1)
        return tuple(encoded)

    def principal_quantum_number(electronic_structure):
        pattern = r'(\d+)([a-z]*)'
        match = re.findall(pattern, electronic_structure)
        if match:
            principal_quantum_number = int(match[0][0])
            return principal_quantum_number
        else:
            return None
        

class PeriodicTable:
    def __init__(self, filename='periodic_table.csv'):

        self.MAXZ = 86 # Only use elements until Radon 

        self.table = {} # Fill up the periodic table 
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                if int(line[0]) <= self.MAXZ:
                    e = Element(int(line[0])) # atomic number Z (atom rank order 1-86)
                    e['symbol'], e['name'] = line[1], line[2] # symbol and name
                    e['n'] = Element.principal_quantum_number(line[5]) # quantum number n (row)
                    e['state'] = e.one_hot_encode(self.MAXZ)
                    self.table[e['Z']] = e # Add element to the periodic table
                else: continue
        
    def __getitem__(self, state): # States are one-hot encoded representations of elements
        return self.table[state]
        
    def get_state_z(self, atomic_number):
        matching_states = [state for state, element in self.table if element['z'] == atomic_number]
        if matching_states:
            return matching_states[0]
        else: return None

    def next_element(self, z):
        element = self.table[z]
        if not element or element['Z'] >= self.MAXZ: return None  # if Radon (last element)
        return element['Z']+1
    
    def previous_element(self, z):
        element = self.table[z]
        if not element or element['Z'] > self.MAXZ: return None 
        if element['Z'] > 1:  # if not  Hydrogen
            return element['Z']-1
        else:
            return None  # No previous element

    def element_above(self, z):
        element = self.table[z]
        if not element or element['Z'] > self.MAXZ: return None 

        # Line 6: no element above for lanthanoids
        if element['n'] == 6:
            if element['Z'] >= 57 and element['Z'] <= 71:
                return None
            elif element['Z'] >= 72:
                return element['Z']-32
            else:
                return element['Z']-18
            
        # Line 5: 
        if element['n'] == 5:
            return element['Z']-18

        # Line 4: no element above for transition metals
        if element['n'] == 4:
            if element['Z'] >= 21 and element['Z'] <= 30:
                return None
            elif element['Z'] >= 31:
                return element['Z']-18
            else:
                return element['Z']-8
    
        # Line 3: 
        if element['n'] == 3:
            return element['Z']-8

        # Line 2: no element above except for Li and Ne
        if element['n'] == 2:
            if element['Z'] == 3:
                return 1
            elif element['Z'] == 10:
                return 2
            else: return None
        
        return None  # No element above for line 1 

    def element_below(self, z):
        element = self.table[z]

        if element['n'] > 5: return None 
        
        elif element['n'] == 5: 
            if element['Z'] <= 39: return element['Z']+18
            else: return element['Z']+32

        elif element['n'] == 4: 
            return element['Z']+18
            
        elif element['n'] == 3: 
            if element['Z'] <= 12: return element['Z']+8
            else: return element['Z']+18

        elif element['n'] == 2: 
            return element['Z']+8
        
        elif element['n'] == 1: 
            if element['Z'] == 1: return 3
            elif element['Z'] == 2: return 10
            else: return None  

        else: return None


### TESTS ###

mendeleyev = PeriodicTable('pubchem.csv')
 # NOTE: Download from https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV?response_type=display


print(mendeleyev[1]['name'])
print(mendeleyev[6]['symbol'])
print(mendeleyev[36]['n'])
print(mendeleyev[mendeleyev.element_below(6)]['name'])
print(mendeleyev[mendeleyev.element_above(32)]['name'])
print(mendeleyev[mendeleyev.next_element(13)]['name'])
print(mendeleyev[mendeleyev.previous_element(15)]['name'])


### Fill in the CSV file ###

# From OCP20 dataset: https://next-gen.materialsproject.org/catalysis    

E_ads_OH2 = { 
    'Ag': -0.092,
    'Al': -1.057,
    'As': -0.022,
    'Au': -0.118,
    'B': -2.284,
    'Bi': -0.056,
    'C': -7.851,
    'Ca': -4.414,
    'Cr': -2.276,
    'Cs': -0.478,
    'Cu': -0.207,
    'Fe': -9.081,
    'Ga': -1.482,
    'Ge': -1.811,
    'Hf': -4.439,
    'Hg': -0.132,
    'In': -0.071,
    'Ir': -0.018,
    'K': -0.484,
    'Mn': -0.879,
    'Mo': -2.103,
    'Na': -0.472,
    'Nb': -3.25,
    'Ni': -0.666,
    'Os': -0.581,
    'Pb': -0.103,
    'Pd': -0.072,
    'Pt': -0.027,
    'Rb': -0.39,
    'Re': -1.819,
    'Rh': -0.389,
    'Ru': -0.802,
    'Sb': -0.06,
    'Sc': -3.282,
    'Si': -2.359,
    'Sn': -1.735,
    'Sr': -2.639,
    'Ta': -2.834,
    'Tc': -1.277,
    'Te': -0.13,
    'Ti': -0.771,
    'Tl': -0.082,
    'V': -3.526,
    'W': -1.911,
    'Y': -3.31,
    'Z': -6.78
}

# Path to the input and output files
input_file = 'pubchem.csv'
output_file = 'periodic_table.csv'

# Open the input file for reading and output file for writing
with open(input_file, 'r') as file_in, open(output_file, 'w', newline='') as file_out:
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)

    # Add the "E" column header
    header = next(reader)
    header.append('E_ads_OH2')
    writer.writerow(header)

    # Iterate through each line in the input file
    count = 0
    for line in reader:
        if count >= 86: continue
        if len(line) >= 3:
            element = line[1]
            if element in E_ads_OH2:
                value = E_ads_OH2[element]
                line.append(value)
        writer.writerow(line)
        count += 1

print("CSV file processing complete.")