import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse


def power_law_samples(n, min_value, max_value, exponent):
    r = np.random.uniform(0, 1, n)
    return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--m0', type=int, default=5, help='number of nodes in initial network')
    parser.add_argument('--num_nodes', type=int, default=100, help='number of nodes in final network')
    parser.add_argument('--m_sample', type=str, default='linear_random', help='method to get number of edges to add for a new node')
    parser.add_argument('--m', type=int, default=3, help='number of edges to add for a new node')
    parser.add_argument('--m_lower', type=int, default=1, help='lower bound of number of edges to add')
    parser.add_argument('--m_upper', type=int, default=5, help='upper bound of number of edges to add')
    parser.add_argument('--exponent', type=float, default=-2.5, help='exponent of power law distribution of number of edges to add')
    args = parser.parse_args()
    
    
    # hyperparameters
    m0 = args.m0
    num_nodes = args.num_nodes

    m_sample = args.m_sample
    m_const = args.m
    m_upper = args.m_lower
    m_lower = args.m_upper
    m_exponent = args.exponent

    # initialization
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 1)]) # , (3, 4), (4, 1)

    # Grow the network
    for i in range(m0, num_nodes):
        G.add_node(i)
        potential_nodes = list(set(G.nodes()) - {i})
        if m_sample == 'linear_random':
            m = np.random.randint(m_lower, m_upper)
        elif m_sample == 'power_law':
            m = int(power_law_samples(1, m_lower, m_upper, m_exponent))
        elif m_sample == 'constant':
            m = m_const
        targets = set(np.random.choice(potential_nodes, size=m, replace=False))
        while len(targets) < m:
            if sum(G.in_degree(n) for n in potential_nodes) == 0:
                targets = set(np.random.choice(potential_nodes, size=m, replace=False))
            else:
                probs = [G.in_degree(n) for n in potential_nodes]
                probs = [p/sum(probs) for p in probs]
                targets.add(np.random.choice(potential_nodes, p=probs))

        for new_edge in targets:
            G.add_edge(i, new_edge)

    # Plot the network
    labels = {node: G.in_degree(node) for node in G.nodes()}
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=500, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title('Directed Network with node in-degrees')
    plt.show()

    # Plot the in-degree distribution
    degrees = [G.in_degree(n) for n in G.nodes()]
    values, counts = np.unique(degrees, return_counts=True)
    plt.figure(figsize=(10,6))
    plt.bar(values, counts, width=0.80, color='b')
    plt.title('In-Degree Distribution')
    plt.ylabel('Count')
    plt.xlabel('In-Degree')
    plt.show()


if __name__ == '__main__':
    main()