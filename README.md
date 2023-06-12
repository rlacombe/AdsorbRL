# AdsorbRL: Deep Reinforcement Learning for Inverse Catalyst Design

In this project, we focus on inverse catalyst design, an inverse design problem where we seek to identify high-performance catalysts for heterogeneous thermal or electro-chemical catalysis with applications for climate solutions.

Specifically, **we use Deep Reinforcement Learning (DRL) to train an agent to navigate the vast space of possible materials and adsorbates, and identify catalyst candidates of interest**. A key descriptor of catalytic activity is the energy with which the reagent species binds to surface of the catalyst. We  use offline RL on the [Materials Project]() and [Open Catalyst 2020]() datasets of adsorption energies to train an RL agent to identify catalysts which bind the strongest (lowest adsorption energy) with select target adsorbates of importance for the clean energy transition.

We specifically worked with a DQN network powered by the Deepmind [Acme framework]() under [TensorFlow]().


### Getting Started

1.  Data
Our datasets are available for download under the `data/` directory.

2.  Model
  1.  To get your model running, first setup a compatible server.  We utilized AWS C4.4xlarge ec2 instances with the Deep Learning AMI GPU Pytorch 1.13.1 (Ubuntu 20.04)
  2.  Down load this repository via git
  3.  Run the following scripts

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

  4. Add the data to the code repository
  5. Choose your appropriate Branch.
      A. Main for main DQN experiments
      C. HER/PER experiments for WIP HER/PER Research
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
\ run_sas_dqn.py -> train and run a DQN in our sas_env.py state-actions environment
\ run_periodic_q.py -> train and run a Q-learning model on the periodic table environment
\ run_periodic_dqn.py -> train and run a DQN model on the periodic table environment
\ run_multi_objective.py -> multi objective script: train and run a multi objective model

Environments:
\ env.py -> baseline env: baseline Environment
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
│    └─ get_materials_project_data.ipynb -> notebook for data retrieval
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

We explore catalysts associated with the following 6 adsorbates, which are of major significance for sustainable development
1.  OH2: adsorbed water, important for green H2 generation through electrolysis, and for all aqueous electro-chemistry and thermal catalysis (e.g. reverse water gas shift)
2. CH4: adsorbed methane, of importance for direct air capture of natural gas and production of hydrogen from fossil fuels with CO2 capture;
3.  N2: adsorbed molecular nitrogen, of the highest importance for ammonia production (which feeds about 1/3 or humanity), and any direct capture process in atmospheric conditions;
4.  NH3: adsorbed ammonia, very important for ammonia production for fertilizers, as well as ammonia-based energy and feedstock production;
5.  CH2: adsorbed methylene group, important for electro-catalysis of production of ethylene, a key building block of the modern chemicals industry;
 6.  OH: adsorbed hydroxyl group. Ubiquitous in aqueous electro-chemistry, hydroxyl groups are active centers in many important catalytic reactions.

For each adsorbate, the Materials Project Catalysis Explorer provides a set of properties including the formula, bulk formula, adsorption energy, and miller indices (which describe a particular lattice structure for the material). For our select adsorbates there are between 5,000 and 10,000 different catalysts each.

We then filter these for the set of unique up to 3-element catalysts by selecting the lowest adsorption energy catalysts among candidate bulk formulae and miller indices. This reduces our set to between 2,000–3,000 catalysts per adsorbate, for a total of approximately 8,000 total unique catalysts in our full dataset for the 6 adsorbates of interest. We find a set of 55 elements comprising these catalysts and represent our state as the one-hot vector of length 55 where the presence of an element in the catalyst is indicated by a ’1’ in the corresponding dimension.
