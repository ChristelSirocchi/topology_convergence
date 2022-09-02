import random
import numpy as np

###########################################################################################################################
# SELECT NEIGHBOUR

# random selection
def random_selection(self):
    return random.choice(self.neighbours)

# ordered selection
def ordered_selection(self):
    # select nodes in order
    return self.neighbours[self.move_count%len(self.neighbours)]

# degree selection
def degree_selection(self):
    # get neighbour with weighted probability
    return np.random.choice(self.neighbours, 1, p = self.degree_p)[0]

# distance selection
def distance_selection(self):
    return np.random.choice(self.neighbours, 1, p = self.distance_p)[0]

###########################################################################################################################
# INTERACTION

# define functions for interaction
def simple_mean(self, neigh):
    # calculate the mean of the two values
    new_value = (neigh.value + self.value)/2
    # update values
    neigh.value = new_value
    self.value = new_value
    # return updated nodes
    return self, neigh

###########################################################################################################################
# TIME
  
def next_poisson(self, paramt):
    return ss.poisson(paramt).rvs() + 1

def next_expo(self, paramt):
    return random.expovariate(1 / paramt) 


#def next_move_d(self):
#    return ss.poisson(5).rvs() + 1

#def next_move_c(self):
#    return random.expovariate(1 / 5) 
"""
def convergence(self, neigh):
    # update own value
    self.value = self.value + cp*(neigh.value - self.value)
    # update other value
    neigh.value = neigh.value + cp*(self.value - neigh.value)
    # return updated nodes
    return self, neigh

def dual_convergence(self, neigh):
    # check condition for high convergence
    if (np.abs(self.value - neigh.value) < bdp*std):
        # update own value
        self.value = self.value + hcp*(neigh.value - self.value)
        # update other value
        neigh.value = neigh.value + hcp*(self.value - neigh.value)
    else:
        # update own value
        self.value = self.value + lcp*(neigh.value - self.value)
        # update other value
        neigh.value = neigh.value + lcp*(self.value - neigh.value)    
    # return updated nodes
    return self, neigh

def bounded_confidence(self, neigh):
    # check condition for high convergence
    if (np.abs(self.value - neigh.value) < bdp*std):
        # update own value
        self.value = self.value + cp*(neigh.value - self.value)
        # update other value
        neigh.value = neigh.value + cp*(self.value - neigh.value) 
    # return updated nodes
    return self, neigh
"""










