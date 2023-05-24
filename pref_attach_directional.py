import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

m0 = 5
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])  # Start with a small cycle

m = 2
for i in range(m0, 100):  # We want 100 nodes in the end
    G.add_node(i)  # Add new node
    targets = set(np.random.choice(G.nodes(), size=m, replace=False))  # Initially, link to m nodes chosen uniformly at random
    while len(targets) < m:  # If targets set is less than m
        if sum(G.in_degree(n) for n in G.nodes()) == 0:  # To avoid ZeroDivisionError
            targets = set(np.random.choice(G.nodes(), size=m, replace=False))  # Link to m nodes chosen uniformly at random
        else:
            probs = [G.in_degree(n) for n in G.nodes()]  # List of in-degree of each node
            probs = [p/sum(probs) for p in probs]  # Normalize the probabilities
            targets.add(np.random.choice(G.nodes(), p=probs))  # Add a node chosen with preferential attachment

    for new_edge in targets:
        G.add_edge(i, new_edge)  # Add edge to the chosen node

labels = {node: G.in_degree(node) for node in G.nodes()}
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=500, arrows=True)
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('Directed Network with node in-degrees')
plt.show()

degrees = [G.in_degree(n) for n in G.nodes()]
values, counts = np.unique(degrees, return_counts=True)

plt.figure(figsize=(10,6))
plt.bar(values, counts, width=0.80, color='b')
plt.title('In-Degree Distribution')
plt.ylabel('Count')
plt.xlabel('In-Degree')
plt.show()
