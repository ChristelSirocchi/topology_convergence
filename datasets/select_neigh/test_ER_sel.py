# import custom libraries
from convergence_ABM import *
from graph_generators import *

# distribution parameters
param1, param2 = 0, 1

config_param_c = {# "graph" : G,
               # "features" : features,
               # "selection" : random_selection, # select neighbours randomly
               "interaction" : simple_mean, # calculate the average at each interaction
               "next_move" : next_expo, # random expovariate
               "max_time" : 50, # total simulation time
               "log_interval" : 1, # logging interval
               "event_logger" : False, # log all events
               "time_logger" : True # log values at intervals
               }
# set size
s = 500
# generate distribution
V0 = sp.random.normal(param1, param2, s) 
# generate features
features = [AgentFeatures(V0[i], 1, 1) for i in range(s)]
# update features in configuration
config_param_c["features"] = features
    
# set average degree node 
Avgs = list(range(6,60,1))
# set interaction type
Ints = [random_selection, ordered_selection, degree_selection, distance_selection]
Ints_name = ["random", "ordered", "degree", "distance"]
results = {}

i = 0
# for (a, i) in itertools.product(*[[Avgs],[Ints]]):
for a in Avgs:    
    # calculate probability based on average degree
    p = a/(s-1)
    # generate graph
    G = get_connected_erdos(s, p)
    # update graph in configuration
    config_param_c["graph"] = G
    for int_n, int_f in enumerate(Ints):
        config_param_c["selection"] = int_f
        slopes = []
        for r in range(100):
            random.shuffle(features)
            # update features in configuration
            config_param_c["features"] = features
            # generate model
            sim = EventDrivenNetworkModel(config_param_c)
            # evolve model and calculate error
            err, m = sim.run_simulation()
            slopes.append(m)
        # save results
        results[i] = {"size": s, "p" : p, "avg": a, "slope" : np.mean(slopes), "interaction" : Ints_name[int_n]} 
        i += 1
    # inform user
    print(str(a) + " done")

# put in dataframe
df = pd.DataFrame(results).T

df.to_csv("ER_sel.csv")
