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
        encoded[self.data['Z']-1] = 1.0 # NOTE: Z = index + 1 (since Z starts at 1)
        return encoded

    def principal_quantum_number(electronic_structure):
        pattern = r'(\d+)([a-z]*)'
        match = re.findall(pattern, electronic_structure)
        if match:
            principal_quantum_number = int(match[0][0])
            return principal_quantum_number
        else:
            return None
        

class PeriodicTable:
    def __init__(self, filename='periodic_table.csv', max_z=86):

        self.MAXZ = max_z # 86: only elements up to Radon 

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

                    try:
                        e['E_ads_OH2'] = float(line[17])
                    except (ValueError, IndexError):
                        e['E_ads_OH2'] = None

                    self.table[e['Z']] = e # Add element to the periodic table
                else: continue
        
    def __getitem__(self, state): # States are one-hot encoded representations of elements
        return self.table[state]
        
    def z_to_state(self, atomic_number):
        matching_states = [state for state, element in self.table if element['z'] == atomic_number]
        if matching_states:
            return matching_states[0] # find state matching Z
        else: return None

    def state_to_z(self, state):
        return np.argmax(state) + 1 # find argmax of state -> Z

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
