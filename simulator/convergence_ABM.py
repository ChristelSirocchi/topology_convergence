from random import random
from types import MethodType
from simpy import Environment

import time, enum, math
import pandas as pd

from graph_metrics import *
from interaction_methods import *


class AgentFeatures():
    """Base class to define agent properties"""
    def __init__(self, value, state, speed): 
        self.value = value
        self.state = state
        self.speed = speed
        # self.confidence = confidence
        # self.latency = latency

class Agent():
    """Base class for a model agent."""
    def __init__(self, unique_id, model, features, select_neighbour, interact, next_move_time, event_logger):
        # set id number
        self.unique_id = unique_id
        # save model(environment)
        self.model = model
        # move count
        self.move_count = 0
        # define function to select neighbour
        self.select_neighbour = MethodType(select_neighbour, self)
        # define function to interact with neighbour
        self.interact = MethodType(interact, self)
        # get function to plan next move
        self.next_move_time = MethodType(next_move_time, self) 
        # set features
        self.value = features.value
        self.state = features.state
        self.speed = features.speed
        # plan next move
        self.next_move = self.next_move_time(self.speed)
        # time of the last move
        self.last_move = 0
        # event_logger
        self.event_logger = event_logger
        # get neighbours indices (only once in static networks)
        self.neighbours = self.get_neighbours()
        # get neighbours degree
        self.degree_p = self.get_degrees()
        # get neighbours distance
        self.distance_p = self.get_distances()
    
    # get neighbouring nodes
    def get_neighbours(self):
        # get neighbouring index
        neigh_id = list(self.model.G.neighbors(self.unique_id))
        # get neighbouring nodes
        # neigh_list = [self.model.G.nodes[_]['agent'] for _ in agents]
        # shuffle list to reduce chance of syncronization
        random.shuffle(neigh_id)
        return neigh_id

    def get_degrees(self):
        # get all degrees
        degrees = [self.model.G.degree[n] for n in self.neighbours]
        # normalise by max degree
        return [d/sum(degrees) for d in degrees]

    def get_distances(self):
        dist = [1/(len(list(nx.common_neighbors(self.model.G, self.unique_id, n))) + 1) for n in self.neighbours]
        return [d/sum(dist) for d in dist]

    def update_neighbours(self):
        self.neighbours = self.get_neighbours()
        self.degree_p = self.get_degrees()
        self.distance_p = self.get_distances()

    # add statistics to dictionary
    def log_event(self, neigh):
        # generate unique name for the dictionary entry
        key_name = "Agent" + str(self.unique_id) + "_" + str(self.move_count)
        # add entry to model
        self.model.event_log[key_name] = [self.model.get_time(), self.unique_id, neigh.unique_id, self.value]

    # make a move
    def play(self):  
        # get neighbours -- not necessary for static networks
        # self.neighbours = self.update_neighbours()     
        # select neighbours
        neigh_id = self.select_neighbour()
        neigh = self.model.G.nodes[neigh_id]['agent']
        # if not active, activate neighbour -- not necessary for static networks
        # if neigh.state == 0:
        #    neigh.activate()
        # interact
        self, neigh = self.interact(neigh)
        # add move
        self.move_count += 1
        # log only if needed
        if self.event_logger:
            # add statistic
            self.log_event(neigh)
  
    def activate(self):
        pass

class EventDrivenAgent(Agent):
    """Base class for a model agent."""
    def __init__(self, unique_id, model, value, select_neighbour, interact, next_move_time, event_logger):
        super().__init__(unique_id, model, value, select_neighbour, interact, next_move_time, event_logger)
        # register event to the environment
        self.model = model
        # set environment
        self.env = model.env
        # set condition flag for activation
        self.condition_flag = self.env.event()
        # initialize every time an instance of the agent is created
        self.action = self.env.process(self.run()) 
        # if activation
        if self.state == 1:
            self.condition_flag.succeed()

    def run(self):
        while True:
            # wait to be activated
            yield self.condition_flag
            # wait that amount of time
            yield self.env.timeout(self.next_move)
            # make that move
            self.play()
            # get next move
            self.next_move = self.next_move_time(self.speed)

    def activate(self):
        self.state = 1
        # update last move time
        self.last_move = self.model.get_time()
        # activate
        self.condition_flag.succeed()


class NetworkModel():
    """Base class for a network model"""

    def __init__(self, config_param):
        # get graph
        self.G = config_param["graph"]   
        # add features
        self.features = config_param["features"]
        # get initial values
        self.initial_values = [f.value for f in self.features]
        # get current values
        self.current_values = self.initial_values
        # maximum simulation time
        self.until = config_param["max_time"]
        # logging interval
        self.log_interval = config_param["log_interval"]
        # calculate graph properties
        self.stats_net = {}
        # calculate node properties
        self.stats_node = {}
        # record data at intervals 
        self.event_log = {}
        # record events
        self.time_log = {}
        # convergence metrics
        self.metrics = {}

    # calculate graph and node properties
    def get_graph_properties(self):
        # calculate graph metrics
        self.stats_net, self.stats_node = graph_metrics(self.G)
        return self.stats_net, self.stats_node

    def get_nodes(self):
        return [self.G.nodes[_]['agent'] for _ in self.G.nodes]

    def run_simulation(self):
        pass

    def get_time(self):
        pass

    def get_values(self):
    # update nodes current values
        self.current_values = [n.value for n in self.get_nodes()]
        return self.current_values

    # method to retrieve time logs as pandas dataframe
    def get_event_log(self):
        df = pd.DataFrame(self.event_log, index = ["Time","AgentID1","AgentID2","Value"]).T
        df = df.astype({"AgentID_A": 'int32', "AgentID_B": 'int32'})
        return df

    # method to retrieve event logs as pandas dataframe
    def get_time_log(self):   
        df = pd.DataFrame(self.time_log, index = ["Agent_"+ str(i) for i in range(self.G.number_of_nodes())]).T
        return df

    # calculating global error
    def calculate_error(self):
        # get time log if not previously calculates
        df = self.get_time_log()
        # calculate Boyd error
        err = df.apply(lambda x: error_boyd(x, self.initial_values), axis = 1)
        # generate time points for regression
        x = np.linspace(0, self.until - self.log_interval, int(self.until/self.log_interval))
        # calculate contraction rate in the second half of the simulation
        m, c = np.polyfit(x[int(len(x)/2):], np.log(err[int(len(x)/2):]), 1)
        return err, np.abs(m)

    # calculating individual error
    def calculate_indiv_error(self):
        # get time log if not previously calculates
        df = self.get_time_log()
        # calculate Boyd error
        error_indiv = df.apply(lambda x: error_boyd(x, x[0]), axis = 0)
        return error_indiv

    """
    # plotting functions
    def plot_graph(self, title, activate):
        # plot the graph (color of nodes corresponds to value)
        if activate:
            plot_model_activate(self, title)
        else:
            plot_model(self, title)

    def plot_timeseries(self, title):
        # plot values over time
        plot_evolution(self, title)

    def plot_interactions(self, title):
        # plot events at the time they are happening
        plot_events(self, title)

    def plot_heatmap(self, title):
        # plot change of values over time as heatmap
        plot_evolution_heatmap(self, title)

    """

class EventDrivenNetworkModel(NetworkModel):
    # initialization grid
    def __init__(self, config_param):
        super().__init__(config_param) 
        # Set-up environment and graph
        self.env = Environment() #simpy.Environment()  

        # Create agents
        for i in self.G.nodes():    
            # create all agents
            a = EventDrivenAgent(i, self, self.features[i], config_param["selection"], config_param["interaction"], 
                config_param["next_move"], config_param["event_logger"])
            # assign agent
            self.G.nodes[i]['agent'] = a

        # check if time logger is needed    
        if config_param["time_logger"]:
            # Create logger
            self.logger = EventDrivenLogger(self, config_param["log_interval"])

    def run_simulation(self):
        # Run trial
        self.env.run(until=self.until)
        return self.calculate_error()  

    def get_time(self):
        return self.env.now

class Logger():
    def __init__(self, model, logging_interval):
        # get interval
        self.interval = logging_interval
        # get model
        self.model = model

class EventDrivenLogger(Logger):
    def __init__(self, model, logging_interval):
        super().__init__(model, logging_interval) 
        # set environment
        self.env = self.model.env
        # initialize process
        self.action = self.env.process(self.run())

    def run(self):
        while True:
            # get data in nodes and save in dictionary
            self.model.time_log[self.env.now] = self.model.get_values()
            # pause
            yield self.env.timeout(self.interval)

"""
# run 100 times and store results
def run_batch_complete(index):
    # update parameters
    config_param_c["features"] = [AgentFeatures(values1[i], states[i], speeds[i]) for i in range(N)]
    # generate model
    sim_complete = EventDrivenNetworkModel(config_param_c)
    # evolve model
    sim_complete.run_simulation()
    # get results
    df = sim_complete.get_time_log()
    # run name
    name = "Run_" + str(index)
    # save metrics
    complete_500[name] = df.apply(lambda x: group_div(x), axis = 1)
    # inform user
    print(str(index) + " done")
    return complete_500


if __name__ == '__main__':

    pool = Pool(processes = 20)

    pool.map(run_batch_complete, index)

"""
