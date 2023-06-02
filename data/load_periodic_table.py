import periodic_table

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