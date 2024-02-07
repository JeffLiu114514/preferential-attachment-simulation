import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

m0 = 5  # Initial number of nodes
G = nx.complete_graph(m0)

m = 2  # Number of edges to attach from a new node to existing nodes
for i in range(m0, 100):
    probs = [G.degree(n) for n in G.nodes()]
    probs = [p/sum(probs) for p in probs]
    new_edges = np.random.choice(G.nodes(), size=m, replace=False, p=probs)
    G.add_node(i)
    for new_edge in new_edges:
        G.add_edge(i, new_edge)
        
# Visualize the graph with node labels indicating the degree
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False)
#labels = nx.get_node_attributes(G, 'degree')
labels = {node: G.degree(node) for node in G.nodes()}
for key in labels.keys():
    labels[key] = G.degree(key)
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('Network with node degrees')
plt.show()

degrees = [G.degree(n) for n in G.nodes()]
values, counts = np.unique(degrees, return_counts=True)

plt.figure(figsize=(10,6))
plt.bar(values, counts, width=0.80, color='b')
plt.title('Degree Distribution')
plt.ylabel('Count')
plt.xlabel('Degree')
plt.show()
