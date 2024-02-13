import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def power_law_samples(n, min_value, max_value, exponent):
    r = np.random.uniform(0, 1, n)
    return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))

def hierarchical_clustering(dist_matrix, threshold=2.0):
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    return fcluster(linkage_matrix, t=threshold, criterion='distance')

# Parameters
n_initial = 5
new_nodes = 195
new_server_probability = 0.05
in_relay_edge_proba = 0.9

# Initialize graph and servers
G = nx.DiGraph()
G.add_edges_from([(n, (n+1) % n_initial) for n in range(n_initial)])

servers = {i: 0 for i in range(n_initial)}

# Grow the network
for i in range(n_initial, n_initial + new_nodes):
    G.add_node(i)
    if np.random.rand() < new_server_probability:
        m = int(power_law_samples(1, 1, 5, -2.5))
        new_server = max(servers.values()) + 1
        for j in range(i, i+m):
            G.add_node(j)
            servers[j] = new_server
            targets = np.random.choice(list(set(G.nodes()) - {j}), size=np.random.randint(1, 5), replace=False)
            G.add_edges_from([(j, target) for target in targets])
        i += m
    else:
        m = int(power_law_samples(1, 1, 5, -2.5))
        server_id = np.random.choice(list(set(servers.values())))
        servers[i] = server_id
        server_nodes = [node for node, server in servers.items() if server == server_id and node != i]
        other_nodes = [node for node, server in servers.items() if server != server_id and node != i]
        for _ in range(m):
            if np.random.rand() < in_relay_edge_proba and server_nodes:
                targets = np.random.choice(server_nodes, size=1, replace=False)
            elif other_nodes:
                targets = np.random.choice(other_nodes, size=1, replace=False)
            else:
                continue
            G.add_edge(i, targets[0])

# Draw the user-server network
labels = {node: G.in_degree(node) for node in G.nodes()}
colors = [servers[node] for node in G.nodes()]

nodes_by_server = defaultdict(list)
for node, server in servers.items():
    nodes_by_server[server].append(node)

pos = {}
x_shift = 0
y_shift = 0
scale = 3
for server, nodes in nodes_by_server.items():
    subgraph = G.subgraph(nodes)
    pos_subgraph = nx.spring_layout(subgraph, seed=42)
    for k, v in pos_subgraph.items():
        v[0] += x_shift
        v[1] += y_shift
    pos.update(pos_subgraph)
    x_shift += scale
    if x_shift >= scale * np.sqrt(len(nodes_by_server)):
        x_shift = 0
        y_shift -= scale

plt.figure(figsize=(10, 6))
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.get_cmap('rainbow'))
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('User-Server Network with node in-degrees')
plt.show()

# Calculate relevance scores between servers
def calc_relevance(servers, graph):
    users_by_server = defaultdict(list)
    for user, server in servers.items():
        users_by_server[server].append(user)

    n_servers = max(servers.values()) + 1
    relevance = np.zeros((n_servers, n_servers))

    for server in range(n_servers):
        for neighbor in G.neighbors(server):
            if servers[neighbor] != server:
                relevance[servers[neighbor], server] += 1

    return relevance

relevance_scores = calc_relevance(servers, G)
print("Relevance scores between servers:\n" + relevance_scores.__str__())

# Perform hierarchical clustering
dist_matrix = 1 / (1 + relevance_scores)
clusters = hierarchical_clustering(dist_matrix, threshold=2.0)

# Create the server-relay network
H = nx.Graph()
relays = {server: cluster for server, cluster in enumerate(clusters)}
for server1, server2 in np.argwhere(relevance_scores > 0):
    if relays[server1] == relays[server2]:
        H.add_edge(server1, server2)



# Draw the server-relay network
color_map = [relays[node] for node in H.nodes()]

pos_H = nx.spring_layout(H, seed=42)
x_shift = 0
y_shift = 0
for relay in set(relays.values()):
    relay_nodes = [node for node in H.nodes if relays[node] == relay]
    subgraph = H.subgraph(relay_nodes)
    pos_subgraph = nx.spring_layout(subgraph, seed=42)
    for k, v in pos_subgraph.items():
        v[0] += x_shift
        v[1] += y_shift
    pos_H.update(pos_subgraph)
    x_shift += 1.0
    if x_shift >= np.sqrt(len(set(relays.values()))):
        x_shift = 0
        y_shift -= 1.0

plt.figure(figsize=(8, 6))
nx.draw(H, pos_H, node_color=color_map, with_labels=True, cmap=plt.get_cmap('rainbow'))
plt.title('Server-Relay Network')
plt.show()
