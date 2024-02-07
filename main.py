import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
import math
import random
from itertools import chain, combinations

from utils import *

n_nodes = 20
user_server_ratio = 0.1
server_clique_ratio = 1
clique_size_expectation = 5
# nodes_initial_ratio = 0.1
# servers_initial_ratio = 0.2
# n_nodes_initial = int(n_nodes * nodes_initial_ratio)
# n_servers_initial = int(n_servers * servers_initial_ratio)
n_servers = int(n_nodes * user_server_ratio)
n_cliques = int(n_servers * server_clique_ratio)

if __name__ == '__main__':
    main()