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

for i in range(500):
    # uniform random size
    s = np.random.randint(100,1000)
    # uniform random probability
    p = np.random.uniform(np.log(s)/s,np.log(s)/s*10,1)[0]
    # generate graph
    G = get_connected_erdos(s, p)
    # calculate average degree node
    avg = np.mean(list(dict(G.degree).values()))
    # generate distribution
    V0 = sp.random.normal(param1, param2, s) 
    # update graph in configuration
    config_param_c["graph"] = G
    # generate features
    features = [AgentFeatures(V0[i], 1, 1) for i in range(s)]
    # save slopes
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
    results[i] = {"size": s, "p" : p, "avg": avg, "slope" : np.mean(slopes)}    
    # inform user
    print(str(i) + " done")

# put in dataframe
df = pd.DataFrame(results).T

df.to_csv("ER_param.csv")
