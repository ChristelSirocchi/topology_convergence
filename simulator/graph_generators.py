import networkx as nx
import random
import math
from FARZ import *

# generate a connected Erdos-Renyi random graph
def get_connected_erdos(N, p):
    while True:
        G = nx.erdos_renyi_graph(n = N, p = p, directed=False)
        if nx.is_connected(G):
            break
    return G

# generate a connected Geometric Random graph
def get_connected_geometric(N, r):
    while True:
        G = nx.random_geometric_graph(N, r)
        if nx.is_connected(G):
            break
    return G

def _random_subset(seq, m):
    # Return m unique elements from seq.
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets

# adapted from NetworkX
def get_powerlaw_cluster_graph(n, a, p):
    m = int(a/2)
    count_p = 0
    count_c = 0
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering."""
    G = nx.empty_graph(m)  # add m initial nodes (m0 in barabasi-speak)
    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = m  # next node is m
    while source < n:  # Now add the other n-1 nodes
        possible_targets = _random_subset(repeated_nodes, m)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        count_p += 1
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < m:  # add m-1 more new links
            if random.random() < p:  # clustering step: add triangle
                neighborhood = [
                    nbr
                    for nbr in G.neighbors(target)
                    if not G.has_edge(source, nbr) and not nbr == source
                ]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = random.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    count_c += 1
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue  # go to top of while loop
            # else do preferential attachment step if above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            count_p += 1
            repeated_nodes.append(target)
            count = count + 1
        repeated_nodes.extend([source] * m)  # add source node to list m times
        source += 1
    return G, count_p, count_c

# generate a community based random graph (adapted from FARZ)
def get_connected_community(s,k,m):
    while True:
        G, node_communities = generate(farz_params={"n":s, #number of nodes
             "k":k, #number of communities
             "m":m, 
             "alpha":0.5, #transitivity
             "gamma":0.5, #degree correlation 
             "beta":0.9, #strength of the community
             "r":1, # 25/1000 or 5/1000
             'q':0.5, # 0.5 or 0.7
             "phi":10, # community size distribution
             "b":0.1, 
             "epsilon":0.0000001, 
             'directed':False, 
             'weighted':False})
        if nx.is_connected(G):
            break
    return G, node_communities

"""
def suffle_edges(G):
    edges = list(G.edges)
    random.shuffle(edges)
    G1 = nx.Graph()
    G1.add_edges_from(edges)
    return G1

"""
