# CS224 AdsorbRL: Deep Reinforcement Learning for Inverse Catalyst Design

Inverse materials design(IMD), a fundamental challenge in materials science, aims to create new materials with specific properties from first principles.  We aim to utilize deep reinforcement learning(DRL) in this space to help assist with catalyst design, an IMD problem, that can be used for production of fuels and feedstocks from reneweable energy, battery gig-factory by-product recycling or the transformation of waste co2 into valuable products.

We specifically worked with a DQN network powered by the Deepmind Acme framework, with tensorflow.

# Getting Started

1.  Data

2.  Model
  1.  To get your model running, first setup a compatible server.  We utilized AWS C4.4xlarge ec2 instances with the Deep Learning AMI GPU Pytorch 1.13.1 (Ubuntu 20.04)
  2.  Down load this repository via git
  3.  Run the following scripts
  ``
  sudo apt-get install python3.8-venv
  python3.8 -m venv env
  source env/bin/activate
  pip install tensorflow
  pip install dm-env
  pip install acme
  pip install dm-acme[jax,tf]
  pip install dm-acme[envs]
  pip install protobuf==3.20.3
  ``
  4. Add the data to the code repository
  5. Choose your appropriate Branch.
      A. Main for main DQN experiments
      B. Periodic for Constrained State/Actions Space: Periodic Table of Elements
      C. HER/PER experiments for WIP HER/PER Research
  6.  Run with proper py file for experiment
      A. python run_dqn.py for baseinline
      B. python run_multi_objective.py for multi objective experimenbts
      C. python run_sas_dqn.py

#Project Structure
\ run_dqn.py -> baseline script: train and run a baseline Model
\ run_multi_objective.py -> multi objective script: train and run a multi objective model
\ run_sas_dqn.py -> sas script: train and run a dqn in a sas env
\ env.py -> baseline env: baseline Environment
\ multi_objective_env.py -> multi objective env: multi objective Environment
\ sas_env.py -> sas env: next_states Environment
\ README.MD -> this document
│
├─ data/ -> data directory
│    └─ graph.py/ -> script for generating visualizations
\

#Data
The dataset is sourced from the Materials Project, a database of materials compilted by the Department of Energy, with pre-computed properties for a large number of materials.  Specifically we use the Catalysis Explorer, an online application that provides pre-computed adsorption energies for various catalysts in the Materials Project under different configurations, sourced from the Open Catalyst Project OCP2020 dataset.  

We explore catalysts associated with the following 6 adsorbates, which are of major significance for sustainable development
1.  OH2: adsorbed water, important for green H2 generation through electrolysis, and for all aqueous electro-chemistry and thermal catalysis (e.g. reverse water gas shift)
2. CH4: adsorbed methane, of importance for direct air capture of natural gas and production of hydrogen from fossil fuels with CO2 capture;
3.  N2: adsorbed molecular nitrogen, of the highest importance for ammonia production (which feeds about 1/3 or humanity), and any direct capture process in atmospheric conditions;
4.  NH3: adsorbed ammonia, very important for ammonia production for fertilizers, as well as ammonia-based energy and feedstock production;
5.  CH2: adsorbed methylene group, important for electro-catalysis of production of ethylene, a key building block of the modern chemicals industry;
 6.  OH: adsorbed hydroxyl group. Ubiquitous in aqueous electro-chemistry, hydroxyl groups are active centers in many important catalytic reactions.

For each adsorbate, the Materials Project Catalysis Explorer provides a set of properties including the formula, bulk formula, adsorption energy, and miller indices (which describe a particular lattice structure for the material). For our select adsorbates there are between 5,000 and 10,000 different
catalysts each.
We then filter these for the set of unique up to 3-element catalysts by selecting the lowest adsorption energy catalysts among candidate bulk formulae and miller indices. This reduces our set to between 2,000–3,000 catalysts per adsorbate, for a total of approximately 8,000 total unique catalysts in our
full dataset for the 6 adsorbates of interest.
We find a set of 55 elements comprising these catalysts and represent our state as the one-hot vector of length 55 where the presence of an element in the catalyst is indicated by a ’1’ in the corresponding dimension. We present our dataset construction setup in figure

#Results

#References


# DataSet
