from periodic_table import PeriodicTable

### TESTS ###

mendeleyev = PeriodicTable('periodic_table.csv')
 # NOTE: Download from https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV?response_type=display


print(mendeleyev[1]['name'])
print(mendeleyev[6]['symbol'])
print(mendeleyev[36]['n'])
print(mendeleyev[mendeleyev.element_below(13)]['name'])
print(mendeleyev[mendeleyev.element_above(13)]['name'])
print(mendeleyev[mendeleyev.next_element(13)]['name'])
print(mendeleyev[mendeleyev.previous_element(13)]['name'])
print(mendeleyev[mendeleyev.element_above(1)]['name'])
