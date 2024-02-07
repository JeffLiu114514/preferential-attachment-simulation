import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from collections import defaultdict
import math
import random
from itertools import chain, combinations

class GenerationWrapper:
    
    def __init__(self, n_nodes, n_servers, n_cliques) -> None:
        self.n_nodes = n_nodes
        self.n_servers = n_servers
        self.n_cliques = n_cliques
        
        self.servers, self.user_to_server = GenerationWrapper.generate_user_server_mapping(n_nodes, n_servers)
        self.cliques, self.server_to_clique = GenerationWrapper.generate_server_clique_mapping_two(self.servers, n_cliques, clique_size_expectation=5)
        
    
    def generate_user_server_mapping(n_nodes, n_servers):
        # each server has at least one user
        servers = defaultdict(list)
        for i in range(n_servers):
            servers[i].append(i)

        # add users to servers by PA only depending on the number of users in the server
        for i in range(n_servers, n_nodes):
            probs = GenerationWrapper.linear_PA_probs(servers)
            
            server_to_join = np.random.choice(range(n_servers), size=1, replace=False, p=probs)
            servers[server_to_join[0]].append(i)

        user_to_server = {}
        # get reverse mapping
        for server_id, user_ids in servers.items():
            for user_id in user_ids:
                user_to_server[user_id] = server_id
                
        return servers, user_to_server


    # second approach: flipping coins
    def generate_server_clique_mapping_two(servers, n_cliques, clique_size_expectation):
        cliques = defaultdict(set)
        server_to_clique = defaultdict(set)

        server_ids = list(servers.keys())
        server_weights = np.array([len(servers[id]) for id in server_ids]).astype(float)
        server_weights /= np.sum(server_weights)
        # denom = sum([len(servers[id]) for id in server_ids])
        # server_weights = [len(servers[id])/denom for id in server_ids]
        
        
        for clique in range (n_cliques):
            for server in server_ids:
                if np.random.rand() < (server_weights[server] * clique_size_expectation):
                    cliques[clique].add(server)
                    server_to_clique[server].add(clique)
            
            #print(f"{clique}th clique")
            
        return cliques, server_to_clique
    
    
    def grow_network():
    
    
    
    @staticmethod
    def linear_PA_probs(target_dict):
        lengths = {key: len(value) for key, value in target_dict.items()}
        denominator = sum(lengths.values())
        return [n/denominator for n in lengths.values()]
    
    @staticmethod
    def quadratic_PA_probs(target_dict):
        lengths = {key: len(value) for key, value in target_dict.items()}
        denominator = sum(math.pow(i, 2) for i in lengths.values())
        return [math.pow(n, 2)/denominator for n in lengths.values()] 
    
    @staticmethod
    def power_law_samples(n, min_value, max_value, exponent):
        r = np.random.uniform(0, 1, n)
        return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))