import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

def power_law_samples(n, min_value, max_value, exponent):
    r = np.random.uniform(0, 1, n)
    return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))


# Parameters
n_initial = 5
new_nodes = 95
in_cluster_edge_proba = 0.9
new_cluster_probability = 0.05

# Start with one initial cluster
G = nx.DiGraph()
G.add_edges_from([(n, (n+1) % n_initial) for n in range(n_initial)]) # initialize with cycle

clusters = {i: 0 for i in range(n_initial)}  # Initial nodes all belong to cluster 0

# Grow the network
for i in range(n_initial, n_initial + new_nodes):
    G.add_node(i)
    if np.random.rand() < new_cluster_probability:  # Form a new cluster
        m = int(power_law_samples(1, 1, 5, -2.5))  # Number of initial nodes in the new cluster
        new_cluster = max(clusters.values()) + 1  # ID of the new cluster
        for j in range(i, i+m):
            G.add_node(j)
            clusters[j] = new_cluster
            targets = np.random.choice(list(set(G.nodes()) - {j}), size=np.random.randint(1, 5), replace=False)
            G.add_edges_from([(j, target) for target in targets])
        i += m  # Skip the nodes that we just added
    else:  # Join an existing cluster
        m = int(power_law_samples(1, 1, 5, -2.5))  # Number of edges to add
        cluster_id = np.random.choice(list(set(clusters.values())))  # Choose a cluster to join
        clusters[i] = cluster_id
        cluster_nodes = [node for node, cluster in clusters.items() if cluster == cluster_id and node != i]
        other_nodes = [node for node, cluster in clusters.items() if cluster != cluster_id and node != i]
        for _ in range(m):
            if np.random.rand() < in_cluster_edge_proba and cluster_nodes:  # Preferential attachment within the cluster
                targets = np.random.choice(cluster_nodes, size=1, replace=False)
            elif other_nodes:  # Preferential attachment to other clusters
                targets = np.random.choice(other_nodes, size=1, replace=False)
            else:
                continue
            G.add_edge(i, targets[0])

# # Prepare the labels (in-degree of each node)
# labels = {node: G.in_degree(node) for node in G.nodes()}

# # Prepare the colors (based on cluster ID)
# colors = [clusters[node] for node in G.nodes()]

# # Create a copy of the graph for layout computation
# G_layout = G.copy()

# # Set the weights of edges for layout computation
# for u, v in G_layout.edges():
#     if clusters[u] == clusters[v]:
#         G_layout[u][v]['weight'] = 10.0  # High weight for edges within clusters
#     else:
#         G_layout[u][v]['weight'] = 0.1  # Low weight for edges between clusters

# # Compute the layout
# pos = nx.spring_layout(G_layout)

# # Draw the graph
# plt.figure(figsize=(10,10))
# nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.get_cmap('rainbow'))
# nx.draw_networkx_labels(G, pos, labels=labels)
# plt.title('Directed Network with node in-degrees')
# plt.show()

# Prepare the labels (in-degree of each node)
labels = {node: G.in_degree(node) for node in G.nodes()}

# Prepare the colors (based on cluster ID)
colors = [clusters[node] for node in G.nodes()]

# Group nodes by clusters
nodes_by_cluster = defaultdict(list)
for node, cluster in clusters.items():
    nodes_by_cluster[cluster].append(node)

print(nodes_by_cluster)
# Compute a separate spring layout for each cluster and adjust the positions
pos = {}

# Determine the number of clusters per row/column in the grid
clusters_per_row = math.ceil(math.sqrt(len(nodes_by_cluster)))

x_shift = 0
y_shift = 0
for i, (cluster, nodes) in enumerate(nodes_by_cluster.items()):
    subgraph = G.subgraph(nodes)
    pos_subgraph = nx.spring_layout(subgraph, seed=42)
    # Adjust positions
    for k, v in pos_subgraph.items():
        v[0] += x_shift
        v[1] += y_shift
    pos.update(pos_subgraph)
    x_shift += 2.0  # Update the x-shift
    if (i + 1) % clusters_per_row == 0:  # If this cluster is the last one in its row
        x_shift = 0  # Reset the x-shift
        y_shift -= 2.0  # Update the y-shift


# Draw the graph
plt.figure(figsize=(10,10))
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.get_cmap('rainbow'))
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('Directed Network with node in-degrees')
plt.show()