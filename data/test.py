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
    'Zr': -0.678
}

values = E_ads_OH2.values()
average = sum(values) / len(values)

print(average)