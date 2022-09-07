# import custom libraries
from convergence_ABM import *
from graph_generators import *

# distribution parameters
param1, param2 = 0, 1

config_param_c = {# "graph" : G,
               # "features" : features,
               "selection" : random_selection, # select neighbours randomly
               "interaction" : simple_mean, # calculate the average at each interaction
               "next_move" : next_expo, # random expovariate
               "max_time" : 50, # total simulation time
               "log_interval" : 1, # logging interval
               "event_logger" : False, # log all events
               "time_logger" : True # log values at intervals
               }
results = {}

# set size
s = 500
# generate distribution
V0 = sp.random.normal(param1, param2, s) 
# generate features
features = [AgentFeatures(V0[i], 1, 1) for i in range(s)]
# randomize features
features_r = features.copy()
# set average degree node 
Avgs = list(range(6,30,4))

i = 0
for a in Avgs:    
    # calculate probability based on average degree
    p = a/(s-1)
    # generate graph
    G = get_connected_erdos(s, p)
    # update graph in configuration
    config_param_c["graph"] = G  
    for r in range(100):
        # update features in configuration
        config_param_c["features"] = features
        # generate model
        sim = EventDrivenNetworkModel(config_param_c)
        # evolve model and calculate error
        err, m = sim.run_simulation()
        results[i] = {"size": s, "p" : p, "avg": a, "slope" : m, "shuffled" : False} 
        i += 1
        # save results
    for r in range(100):
        random.shuffle(features_r)
        # update features in configuration
        config_param_c["features"] = features_r
        # generate model
        sim = EventDrivenNetworkModel(config_param_c)
        # evolve model and calculate error
        err, m = sim.run_simulation()
        results[i] = {"size": s, "p" : p, "avg": a, "slope" : m, "shuffled" : True} 
        # save results
        i += 1
    # inform user
    print(str(a) + " done")

# put in dataframe
df = pd.DataFrame(results).T

df.to_csv("ER_init.csv")
