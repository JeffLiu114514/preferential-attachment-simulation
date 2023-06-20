import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import math
from collections import defaultdict

n_initial_servers = 5
G = nx.DiGraph()
servers = [(i * 2) for i in range(n_initial_servers)]
users = [(i * 2 + 1) for i in range(n_initial_servers)]
edges = [(servers[i], users[i]) for i in range(n_initial_servers)]
G.add_edges_from(edges)

node_to_server = {user: server for server, user in zip(servers, users)}

new_nodes = 100
new_server_probability = 0.1
new_clique_probability = 0.1

server_cliques = [[server] for server in servers]

H = nx.Graph()

# Add new nodes to the network
for i in range(new_nodes):
    if np.random.rand() < new_server_probability:  # Form a new server
        server_id = max(G.nodes) + 1
        G.add_node(server_id)
        G.add_edge(server_id, i)
        node_to_server[i] = server_id
        servers.append(server_id)
        if np.random.rand() < new_clique_probability:  # Form a new clique
            server_cliques.append([server_id])
        else:  # Join an existing clique
            clique = random.choice(server_cliques)
            clique.append(server_id)
            # Add edges to auxiliary graph
            for server in clique:
                if server != server_id:
                    H.add_edge(server, server_id)
    else:  # Join an existing server
        server_id = np.random.choice(list(set(node_to_server.values())))
        G.add_edge(server_id, i)
        node_to_server[i] = server_id

print(node_to_server)
print(server_cliques)

pos = {}
for server in servers:
    nodes = [node for node, srv in node_to_server.items() if srv == server]
    subgraph = G.subgraph(nodes + [server])
    pos_subgraph = nx.spring_layout(subgraph, scale=0.25, seed=42)
    pos.update(pos_subgraph)

servers_per_row = math.ceil(math.sqrt(len(servers)))
spacing = 2.0
for i, server in enumerate(servers):
    dx = (i % servers_per_row) * spacing
    dy = (i // servers_per_row) * spacing
    for node in [node for node, srv in node_to_server.items() if srv == server] + [server]:
        pos[node] = np.array([pos[node][0] + dx, pos[node][1] + dy])

# Create a mapping of nodes to colors based on server
color_map = plt.cm.get_cmap('rainbow', len(servers))
colors = [servers.index(node_to_server[node]) for node in G.nodes if node not in servers] + [servers.index(node) for node in servers]
sizes = [300 if node in servers else 50 for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, cmap=color_map)
user_server_edges = [(server, user) for server, user in node_to_server.items()]
nx.draw_networkx_edges(G, pos, edgelist=user_server_edges, edge_color='blue')

# Show cliques in auxiliary graph using text labels
labels = {server: f'Clique {i+1}' for i, clique in enumerate(server_cliques) for server in clique}
for server, label in labels.items():
    plt.text(pos[server][0], pos[server][1] + 0.05, label, horizontalalignment='center')

plt.show()
