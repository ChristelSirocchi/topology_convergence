import scipy as sp
import scipy.stats as ss
import numpy as np
import networkx as nx

# calculate error Denantes et al. 2016
def error_norm(values):
    avg = np.mean(values)
    return np.sum((values - avg)**2)**0.5

# calculate error Boyd et al. 2005
def error_boyd(values, V0):
    avg = np.mean(values)
    return np.sum((values - avg)**2)**0.5/np.sum((np.array(V0))**2)**0.5

# calculate nodes and graph metrics
def graph_metrics(G):
    # empty dictionary to store statistics
    stats_net = {}
    stats_node = {}

    # compute distance measures or measures of non centrality (6)
    ecc = list(nx.eccentricity(G).values()) 
    stats_node["ecc"] = ecc        
    stats_net["avg_ecc"] = np.mean(ecc)    # average eccentricity
    stats_net["med_ecc"] = np.median(ecc)  # median eccentricity
    stats_net["std_ecc"] = np.std(ecc)     # standard deviation of eccentricity
    stats_net["diameter"] = np.max(ecc)             # maximum eccentricity (diameter)
    stats_net["radius"] = np.min(ecc)               # minimum eccentricity (radius)
    stats_net["skew_ecc"] = ss.skew(ecc)   # eccentricity skewness
    
    # compute centrality measures (20)
    deg = list(nx.degree_centrality(G).values())
    stats_node["degree_c"] = deg
    stats_net["avg_degree"] = np.mean(deg)  # average degree
    stats_net["std_degree"] = np.std(deg)   # standard deviation of degree
    stats_net["max_degree"] = np.max(deg)   # maximum degree
    stats_net["min_degree"] = np.min(deg)   # minimum degree
    stats_net["skew_degree"] = ss.skew(deg) # degree skewness

    clos = list(nx.closeness_centrality(G).values()) # closeness centrality
    stats_node["clos_c"] = clos
    stats_net["avg_clos"] = np.mean(clos)
    stats_net["std_clos"] = np.std(clos)
    stats_net["max_clos"] = np.max(clos)
    stats_net["min_clos"] = np.min(clos)
    stats_net["skew_clos"] = ss.skew(clos)

    betw = list(nx.betweenness_centrality(G).values()) # betweenness centrality
    stats_node["betw_c"] = betw
    stats_net["avg_betw"] = np.mean(betw)
    stats_net["std_betw"] = np.std(betw)
    stats_net["max_betw"] = np.max(betw)
    stats_net["min_betw"] = np.min(betw)
    stats_net["skew_betw"] = ss.skew(betw)

    try:
        eig = list(nx.eigenvector_centrality(G, max_iter=5000).values()) # eigenvector centrality
        stats_node["eig_c"] = eig
        stats_net["avg_eig"] = np.mean(eig)
        stats_net["std_eig"] = np.std(eig)
        stats_net["max_eig"] = np.max(eig)
        stats_net["min_eig"] = np.min(eig)
        stats_net["skew_eig"] = ss.skew(eig)
    except:
        stats_net["avg_eig"] = np.nan
        stats_net["std_eig"] = np.nan
        stats_net["max_eig"] = np.nan
        stats_net["min_eig"] = np.nan
        stats_net["skew_eig"] = np.nan
        
    # connectivity measures (5)
    clust = list(nx.clustering(G).values()) # clustering
    stats_node["clust_c"] = clust
    stats_net["avg_clust"] = np.mean(clust) # transitivity : nx.transitivity(G)
    stats_net["std_clust"] = np.std(clust)
    stats_net["max_clust"] = np.max(clust)
    stats_net["min_clust"] = np.min(clust)
    stats_net["skew_clust"] = ss.skew(clust)

    # efficiency (5)
    eff = [nx.global_efficiency(G.subgraph(G[v])) for v in G]
    stats_node["eff"] = eff
    stats_net["avg_eff"] = np.mean(eff) # local efficiency : nx.local_efficiency(G)
    stats_net["std_eff"] = np.std(eff)
    stats_net["max_eff"] = np.max(eff)
    stats_net["min_eff"] = np.min(eff)
    stats_net["skew_eff"] = ss.skew(eff)

    # Global efficiency
    stats_net["glb_eff"] = nx.global_efficiency(G)

    # Shannon entropy degree
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    count = np.bincount(degree_sequence)
    stats_net["entropy_degree"] = ss.entropy(count[count != 0] / sum(count)) 

    # Assortativity coefficient
    stats_net["assort_corr"] = nx.degree_pearson_correlation_coefficient(G)
    
    # Average Shortest Path length
    stats_net["avg_short_path"] = nx.average_shortest_path_length(G)    
    
    # Weiner Index
    stats_net["w_ind"] = nx.wiener_index(G)          

    # spectral properties (extreme eigenvalues) (8)
    eig_laplacian = sorted(nx.laplacian_spectrum(G).real, reverse=True)
    stats_net["l1_lapl"] = eig_laplacian[0]
    stats_net["l2_lapl"] = eig_laplacian[1]
    #stats_net["ln_lapl"] = eig_laplacian[-1]
    stats_net["ln_1_lapl"] = eig_laplacian[-2]   # algebraic connectivity : nx.algebraic_connectivity(G, method='lanczos')

    eig_adjecency = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    stats_net["l1_adj"] = eig_adjecency[0]       # spectral radius
    stats_net["l2_adj"] = eig_adjecency[1]
    stats_net["ln_adj"] = eig_adjecency[-1]
    stats_net["ln_1_adj"] = eig_adjecency[-2]
    
    return stats_net, stats_node






# function to find neighbour of neighbour
def get_neigh_neigh(G, node):
    neigh_neigh = {}
    for neigh in G.neighbors(node):
        neigh_neigh[neigh] = [n for n in G.neighbors(neigh)]

# function to remove triangles incident to a node
def remove_triangle(G, node):
    # find neighbours of node
    neigh = [n for n in G.neighbors(node)]
    # for each neighbour
    for n in neigh:
        # find neigh of neigh
        neigh_neigh = [m for m in G.neighbors(n)]  
        # find common neighbours
        common_neigh = set(neigh) & set(neigh_neigh)
        # remove if present
        if len(common_neigh) != 0:
            # get first common neighbour
            common = list(common_neigh)[0]
            # and remove link
            print(node, neigh, common)
            G.remove_edge(node, common)
            break
    return neigh_neigh

# remove all triangles in a graph
def remove_all_triangles(G):
    while sum(nx.triangles(G).values()) > 0:
        # get nodes
        nodes = list(dict(G.degree).keys())
        # get degrees
        degrees = list(dict(G.degree).values())
        # get degree dictionary
        degrees_p = [d/sum(degrees) for d in degrees]
        # find most connected node
        node = np.random.choice(nodes, 1, p = degrees_p)[0]
        # remove a triangle
        remove_triangle(G, node)    
    return G

# check if two nodes are within a given distance from each other
def check_if_close(center, node, d):
    return (np.abs(center["pos"][0] - node["pos"][0]) < d) & (np.abs(center["pos"][1] - node["pos"][1]) < d)

# calculate distance as
def calculate_dist(G):
    neigh_dist = {}
    for n in G.nodes():
        neighbours = G.neighbors(n)
        dist = [1/(len(list(nx.common_neighbors(G, n, m))) + 1) for m in neighbours]
        dist_p = [d/sum(dist) for d in dist]
        neigh_dist[n] = dist_p 
    return neigh_dist
