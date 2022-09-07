# Topological network features determine convergence rate of distributed average algorithms

This repository contains code and data for the manuscript "**Topological network features determine convergence rate of distributed average algorithms**"

It contains two folders:
* simulator
   * convergence_ABM.py: event-driven simulator based on the Python library Simpy
   * graph_generators.py: functions generating connected graphs for the four chosen graph families (Erdos-Renyi, geometric random, small world, scale-free)
   * graph_metrics.py: functions calculating the 48 selected graph metrics (global, local and spectral metrics)
   * interaction_methods.py: functions defining neighbour selection criteria (random, distance, degree, ordered), type of interaction (averaging), time to next move (Poisson)
* datasets: data generated for each graph family and example of test file to generate a similar dataset
   * graph_metric: graph metrics and convergence rates for over 12000 graphs
   * graph_param: convergence rates for graphs generated from random parameters
   * initial_values: convergence rates obtained with and without shuffling the initial values
   * select_neigh: convergence rates for different gossiping algorithms
 
The simulator can be easily adapted to model time-varying topologies, communication delays, heterogeneous agent behaviour (e.g. nodes adopting different interaction rates, time delays or neighbours selection criteria), or game theoretic formulations of agent behaviour. The dataset can be used to develop more accurate regression models or machine learning models for predicting convergence rates from graph features.



