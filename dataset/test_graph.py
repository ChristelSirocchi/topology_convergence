# import custom libraries
import itertools
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
    
# set average degree node 
Avgs = list(range(9,60,1))
Sizes = list(range(200,1000,50))
results = {}

rep = 100    
i = 0

try:
    for (s, a) in itertools.product(*[Sizes, Avgs]):   
        ####### ERDOS RENYI
        # calculate probability based on average degree
        p = a/(s-1)
        # generate graph
        G = get_connected_erdos(s, p)
        # generate features
        V0 = sp.random.normal(param1, param2, s) 
        features = [AgentFeatures(V0[i], 1, 1) for i in range(s)]
        # update graph in configuration
        config_param_c["graph"] = G  
        slopes = []
        for _ in range(rep):
            # generate model
            random.shuffle(features)
            # update features in configuration
            config_param_c["features"] = features
            sim = EventDrivenNetworkModel(config_param_c)
            # evolve model and calculate error
            err, m = sim.run_simulation()
            slopes.append(m)
            # save results
        stats_net, stats_node = sim.get_graph_properties()
        stats_slope = {"graph":"ER", "size": s, "p" : p, "avg": a, "slope" : np.mean(slopes)} 
        stats_slope.update(stats_net)
        results[i] = stats_slope
        i += 1
        ###### SMALL WORLD
        for pr in np.linspace(0,1,11):
            G = nx.connected_watts_strogatz_graph(s, a, pr)
            config_param_c["graph"] = G  
            slopes = []
            for _ in range(rep):
                random.shuffle(features)
                # update features in configuration
                config_param_c["features"] = features
                # generate model
                sim = EventDrivenNetworkModel(config_param_c)
                # evolve model and calculate error
                err, m = sim.run_simulation()
                slopes.append(m)
            # save results
            stats_net, stats_node = sim.get_graph_properties()
            stats_slope = {"graph":"SW", "size": s, "p" : pr, "avg": a, "slope" : np.mean(slopes)} 
            stats_slope.update(stats_net)
            results[i] = stats_slope
            i += 1
        ###### SCALE FREE
        for pc in np.linspace(0,1,11):
            G, count_p, count_c = get_powerlaw_cluster_graph(s, a, pc)
            config_param_c["graph"] = G  
            slopes = []
            for _ in range(rep):
                random.shuffle(features)
                # update features in configuration
                config_param_c["features"] = features
                # generate model
                sim = EventDrivenNetworkModel(config_param_c)
                # evolve model and calculate error
                err, m = sim.run_simulation()
                slopes.append(m)
            # save results
            stats_net, stats_node = sim.get_graph_properties()
            stats_slope = {"graph":"SF", "size": s, "p" : pc, "avg": a, "slope" : np.mean(slopes)} 
            stats_slope.update(stats_net)
            results[i] = stats_slope
            i += 1
        # Geometric random
        r = math.sqrt(a/(math.pi*s))
        # generate graph
        G = get_connected_geometric(s, r)
        config_param_c["graph"] = G  
        slopes = []
        for _ in range(rep):
            random.shuffle(features)
            # update features in configuration
            config_param_c["features"] = features
            # generate model
            sim = EventDrivenNetworkModel(config_param_c)
            # evolve model and calculate error
            err, m = sim.run_simulation()
            slopes.append(m)
        # save results
        stats_net, stats_node = sim.get_graph_properties()
        stats_slope = {"graph":"GR", "size": s, "p" : r, "avg": a, "slope" : np.mean(slopes)} 
        stats_slope.update(stats_net)
        results[i] = stats_slope
        i += 1
        # inform user
        print(str(s) + " "+ str(a) + " done")

finally:
    # put in dataframe
    df = pd.DataFrame(results).T

    df.to_csv("all_graphs.csv")
