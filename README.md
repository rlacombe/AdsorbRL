# AdsorbRL: Deep Reinforcement Learning for Inverse Catalyst Design

In this project, we focus on inverse catalyst design, an inverse design problem where we seek to identify high-performance catalysts for heterogeneous thermal or electro-chemical catalysis with applications for climate solutions.

Specifically, **we use Deep Reinforcement Learning (DRL) to train an agent to navigate the vast space of possible materials and adsorbates, and identify catalyst candidates of interest**. A key descriptor of catalytic activity is the energy with which the reagent species binds to surface of the catalyst. We  use offline RL on the [Materials Project](https://next-gen.materialsproject.org/materials) and [Meta AI Open Catalyst 2020](https://opencatalystproject.org/) datasets of adsorption energies to train an RL agent to identify catalysts which bind the strongest (lowest adsorption energy) with select target adsorbates of importance for the clean energy transition. We develop a DQN network powered by the [Deepmind Acme framework](https://arxiv.org/abs/2006.00979) and train it to navigate the materials dataset with a single or multi-objective target for binding energy minimization (or maximization, respectively).

<img width="890" alt="AdsorbRL" src="https://github.com/rlacombe/AdsorbRL/assets/3156495/e92b01a0-af86-4667-be16-6543babaa6be">



### Getting Started

1.  **Data**:
Our datasets are available for download under the `data/` directory.

2.  **Model**:
Setup a compatible server.  We used AWS C4.4xlarge ec2 instances with the Deep Learning AMI GPU Pytorch 1.13.1 (Ubuntu 20.04).
Clone or fork this repository and run the following scripts:

  ```
  sudo apt-get install python3.8-venv
  python3.8 -m venv env
  source env/bin/activate
  pip install tensorflow
  pip install dm-env
  pip install acme
  pip install dm-acme[jax,tf]
  pip install dm-acme[envs]
  pip install protobuf==3.20.3
  ```

3. Add the data to the code repository
  
4. Choose your appropriate branch.
      A. `main` for main DQN experiments
      B. `HER/PER experiments` for WIP HER/PER Research
   
6.  Run with proper py file for experiment
      A. python run_dqn.py for baseinline
      B. python run_multi_objective.py for multi objective experimenbts
      C. python run_sas_dqn.py
      D. python run_periodic_dqn.py
      E. python run_periodic_q.py


### Project Structure

```
Scripts:
\ run_dqn.py -> baseline: train and run a DQN in env.py baseline environment
\ run_dqn_offlineReward.py: train and run a DQN with offline data and reward shaping for 'invalid' states 
\ run_sas_dqn.py -> train and run a DQN in our sas_env.py state-actions environment
\ run_periodic_q.py -> train and run a Q-learning model on the periodic table environment
\ run_periodic_dqn.py -> train and run a DQN model on the periodic table environment
\ run_multi_objective.py -> multi objective script: train and run a multi objective model

Environments:
\ env.py -> baseline env: baseline Environment
\ env_offlineReward.py: offline Environment with reward shaping
\ periodic_env.py periodic env: periodic table Environment
\ sas_env.py -> sas env: next_states Environment
\ multi_objective_env.py -> multi-objective Environment

Utils:
\ requirements.txt -> install requirements to run the environment
\ explore.py -> define Qlearning agent and epsilon-greedy exploration schedule
\ README.MD -> this document
│
├─ data/ -> data directory
│    └─ graph.py -> graph data analysis and visualization
│    └─ get_unique_elements.ipynb -> decompose bulk formula into their component elements
│    └─ get_materials_project_data.ipynb -> notebook for data retrieval and creation of (s, a, r, s') tuples
│    └─ starOH2/ -> dataset for *OH2 adsorbate
│    └─ starCH2/ -> dataset for *CH2 adsorbate
│    └─ starCH4/ -> dataset for *CH4 adsorbate
│    └─ starH/ -> dataset for *H adsorbate
│    └─ starN2/ -> dataset for *N2 adsorbate
│    └─ starNH3/ -> dataset for *NH3 adsorbate
│    └─ starOH/ -> dataset for *OH adsorbate
│    └─ starOHCH3/ -> dataset for *OHCH3 adsorbate
│
├─ periodic/ -> data and setup directory for the periodic table environment
│    └─ periodic_table.py -> define the PeriodicTable class
│    └─ load_periodic_table.py -> load the periodic table data
│    └─ test_periodic_table.py -> test the PeriodicTable class
│    └─ pubchem.csv -> raw data from PubChem
│    └─ periodic_table.py -> full periodic table data

```

### Data
The dataset is sourced from the Materials Project, a database of materials compilted by the Department of Energy, with pre-computed properties for a large number of materials.  Specifically we use the Catalysis Explorer, an online application that provides pre-computed adsorption energies for various catalysts in the Materials Project under different configurations, sourced from the Open Catalyst Project OCP2020 dataset.  

We explore catalysts associated with the following 6 adsorbates, which are of major significance for clean energy conversion:

1. ⋆OH2: adsorbed water, a key reagent for the Oxygen Evolution Reaction (OER) in H2 generation from water electrolysis, and a key product of the Oxygen Reduction Reaction (ORR) reaction for H2 fuel cells;

2. ⋆OH: adsorbed hydroxyl radical, a key intermediate for OER and ORR, often a rate-limiting step requiring the application of electrochemical over-potentials as its adsorption energy tends to scale linearly with those of ⋆O and ⋆OOH;

3. ⋆CH4: adsorbed methane, of importance for direct air capture of natural gas, and for the CO2 Reduction Reaction (CO2RR) for e.g. methanation of captured carbon dioxide;

4. ⋆CH2: adsorbed methylene radical, another important intermediate for CO2 Reduction Reaction (CO2RR) for the ethylene electro-catalystic production pathway, a key building block of the modern chemicals industry;

5. ⋆N2: adsorbed molecular nitrogen, a key reagent for the Nitrogen Reduction Reaction (NRR) for ammonia production, an essential ingredient of synthetic fertilizers without which half the world population would not be fed;

6. ⋆NH3: adsorbed ammonia, the desired product of Nitrogen reduction (NRR).

For each adsorbate, the Materials Project Catalysis Explorer provides a set of properties including the formula, bulk formula, adsorption energy, and miller indices (which describe a particular lattice structure for the material). For our select adsorbates there are between 5,000 and 10,000 different catalysts each.

We then filter these for the set of unique up to 3-element catalysts by selecting the lowest adsorption energy catalysts among candidate bulk formulae and miller indices. This reduces our set to between 2,000–3,000 catalysts per adsorbate, for a total of 7,386 total unique catalysts in our full dataset for the 6 adsorbates of interest. We find a set of 55 elements comprising these catalysts and represent our state as the one-hot vector of length 55 where the presence of an element in the catalyst is indicated by a ’1’ in the corresponding dimension.

### Paper

Link to paper on arxiv here: [https://arxiv.org/abs/2312.02308](https://arxiv.org/abs/2312.02308).

If you find this helpful please cite as follows:

```
@inproceedings{lacombe2023catalysts,
  title={{AdsorbRL:} Deep Multi-Objective Reinforcement Learning for Inverse Catalysts Design},
  author={Romain Lacombe and Lucas Hendren and Khalid El-Awady},
  booktitle={37th Conference on Neural Information Processing Systems, AI for Accelerated Materials Design Workshop},
  year={2023}
}
```
