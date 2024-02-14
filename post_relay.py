import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
import math
import random
from itertools import chain, combinations

def main():
    '''
    Parameters
    '''
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
    
    n_posts = n_nodes*5
    
    '''
    Graph Generation
    '''
    servers, user_to_server = generate_user_server_mapping(n_nodes, n_servers)
    # H, cliques, server_to_clique = generate_server_clique_mapping_one(servers, n_cliques)
    cliques, server_to_clique = generate_server_clique_mapping_two(servers, n_cliques, clique_size_expectation)
    
    # print("servers to users mapping:" + str(servers))
    # print("user to server mapping:" + str(user_to_server))
    print("cliques to servers mapping:" + str(cliques))
    print(len(cliques.keys()))
    print("server to cliques mapping:" + str(server_to_clique))
    print(len(server_to_clique.keys()))
    
    G, user_visible_universe = grow_network(n_nodes, user_to_server, server_to_clique, cliques, servers)
    # follower_weights = np.array([len(visible_universe) for visible_universe in user_visible_universe])
    # follower_weights = follower_weights / sum(follower_weights)
    follower_weights = []
    for user in range(n_nodes):
        num_followers = len(G.edges(user))
        # print(f'user {user} has {num_followers} followers.')
        follower_weights.append(num_followers)
    follower_weights = np.array(follower_weights).astype(float) + 1
    follower_weights /= sum(follower_weights)
    # print(follower_weights)
    
    # print(user_visible_universe)
    #visualize_user_graph(G, servers, user_to_server)
    
    '''
    Posts
    '''
    post_to_user, user_to_posts = make_posts(n_nodes, n_posts, follower_weights)
    print(user_to_posts)
    
    
def make_posts(n_users, k_posts, follower_weights):
    '''
    spawn posts to users with PA weight by the number of followers
    based on assumption that users with more followers are likely to post more
    '''
    
    post_to_user = {}
    user_to_posts = defaultdict(list)
    posts_owners = np.random.choice(range(n_users), size=k_posts, replace=True, p=follower_weights)
    for i, post_owner in enumerate(posts_owners):
        post_to_user[i] = post_owner
        user_to_posts[post_owner].append(i)
        
    return post_to_user, user_to_posts
    
def visualize_user_graph(G, servers, user_to_server):
    labels = {node: G.in_degree(node) for node in G.nodes()}

    # Prepare the colors (based on cluster ID)
    colors = [user_to_server[node] for node in G.nodes()]

    # Compute a separate spring layout for each cluster and adjust the positions
    pos = {}

    # Determine the number of clusters per row/column in the grid
    servers_per_row = math.ceil(math.sqrt(len(servers)))

    x_shift = 0
    y_shift = 0
    for i, (cluster, nodes) in enumerate(servers.items()):
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, scale=4, weight=None, iterations=10) #, seed=42
        # Adjust positions
        for k, v in pos_subgraph.items():
            v[0] += x_shift
            v[1] += y_shift
        pos.update(pos_subgraph)
        x_shift += 10.0  # Update the x-shift
        if (i + 1) % servers_per_row == 0:  # If this cluster is the last one in its row
            x_shift = 0  # Reset the x-shift
            y_shift -= 10.0  # Update the y-shift
            
    plt.figure(figsize=(16,10))
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=200, width=0.5, edge_vmax=0.5, cmap=plt.get_cmap('rainbow'))
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title('Directed Network with node in-degrees')
    plt.show()


def generate_user_server_mapping(n_nodes, n_servers):
    # each server has at least one user
    servers = defaultdict(list)
    for i in range(n_servers):
        servers[i].append(i)

    # add users to servers by PA only depending on the number of users in the server
    for i in range(n_servers, n_nodes):
        # quadratically proportional to the number of users in the server
        # lengths = {key: len(value) for key, value in servers.items()}
        # denominator = sum(math.pow(i, 2) for i in lengths.values())
        # probs = [math.pow(n, 2)/denominator for n in lengths.values()] 
        
        # linearly
        lengths = {key: len(value) for key, value in servers.items()}
        denominator = sum(lengths.values())
        probs = [n/denominator for n in lengths.values()]
        
        server_to_join = np.random.choice(range(n_servers), size=1, replace=False, p=probs)
        servers[server_to_join[0]].append(i)

    # get reverse mapping
    user_to_server = {}

    for server_id, user_ids in servers.items():
        for user_id in user_ids:
            user_to_server[user_id] = server_id
        
    return servers, user_to_server


'''
Three different types of server clique mapping generation
'''

# first approach: edge-based joining
def generate_server_clique_mapping_one(servers, n_cliques):
    cliques = defaultdict(set)
    server_to_clique = defaultdict(set)

    server_ids = list(servers.keys())
    denom = sum([len(servers[id]) for id in server_ids])
    server_weights = [len(servers[id])/denom for id in server_ids]

    # Initialize the graph
    G = nx.Graph()

    i = 0
    while i < n_cliques:
        print(f"i: {i}")
        selected_servers = np.random.choice(server_ids,size=2,replace=False, p=server_weights)

        # Add nodes and edge to the graph
        G.add_edge(*selected_servers)

        # # Get the connected component (subgraph) that the selected servers belong to
        # for component in nx.connected_components(G):
        #     if set(selected_servers).issubset(component):
        #         connected_component = component
        #         break

        # # Check if the connected component forms a fully connected subgraph (clique)
        # subgraph = G.subgraph(connected_component)
        # n_nodes = subgraph.number_of_nodes()
        # if subgraph.number_of_edges() == n_nodes * (n_nodes - 1) // 2 and n_nodes > 2:
        #     # Try to find whether this fully connected subgraph is a part of a larger fully connected graph
        #     merged_clique = set()
        #     for server in connected_component:
        #         for clique_id in server_to_clique[server]:
        #             merged_clique.update(cliques[clique_id])
        #             del cliques[clique_id]
        #     print(f"merged_clique: {merged_clique}")
            
        #     cliques[i] = merged_clique
        #     for server in merged_clique:
        #         server_to_clique[server] = {i}

        #     i += 1 - len(merged_clique) + 1
        # else:
        #     cliques[i] = set(selected_servers)
        #     for server in selected_servers:
        #         server_to_clique[server].add(i)
        #     i += 1
        
        # Check for any newly formed cliques
        new_clique = None
        for clique in nx.algorithms.clique.find_cliques(G):
            if len(clique) > 2:
                if set(selected_servers).issubset(set(clique)):
                    new_clique = set(clique)
                    break

        if new_clique:
            subgraph = G.subgraph(new_clique)
            n_nodes = subgraph.number_of_nodes()
            if subgraph.number_of_edges() == n_nodes * (n_nodes - 1) // 2 and n_nodes > 2: # fully connected
                merged_clique = set()
                for server in new_clique:
                    for clique_id in server_to_clique[server]:
                        if set(cliques[clique_id]).issubset(new_clique):
                            merged_clique.update(cliques[clique_id])
                            del cliques[clique_id]
                print(f"new_clique: {new_clique}")
                print(f"merged_clique: {merged_clique}")
                # Plot the graph
                nx.draw(G, with_labels=True)
                plt.show()
                cliques[i] = merged_clique
                for server in merged_clique:
                    server_to_clique[server] = {i}

                i += 1 - len(merged_clique) + 1
                
            else:
                cliques[i] = set(selected_servers)
                for server in selected_servers:
                    server_to_clique[server].add(i)
                i += 1
        else:
            cliques[i] = set(selected_servers)
            for server in selected_servers:
                server_to_clique[server].add(i)
            i += 1
            


    # Plot the graph
    nx.draw(G, with_labels=True)
    plt.show()

    return G, cliques, server_to_clique


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


# def generate_server_clique_mapping(servers, n_cliques, n_servers):
#     # each clique has at least one server
#     cliques = defaultdict(list)
#     server_to_clique = defaultdict(list)
    
#     for i in range(n_cliques):
#         cliques[i].append(i)

#     server_ids = list()
#     server_weights = [len(servers[id]) for id in servers.keys()]
    
#     for i in range(n_cliques):
#         # Randomly select two servers with weights proportional to their user counts
#         selected_servers = random.choices(server_ids, weights=server_weights, k=2)

#         for server in selected_servers:
#             cliques[i].append(server)
#             server_to_clique[server].append(i)
            
#     return cliques, server_to_clique
'''
    # spawn cliques among servers by PA only depending on the number of users in the server
    for i in range(n_cliques, n_servers):
        # linearly proportional to the number of users in the server
        lengths = {key: len(value) for key, value in cliques.items()}
        denominator = sum(lengths.values())
        probs = [n/denominator for n in lengths.values()]
        clique_to_join = np.random.choice(range(n_cliques), size=1, replace=False, p=probs)
        cliques[clique_to_join[0]].append(i)

    print("clique to server mapping: \n")
    print(cliques)
    print()
    print("number of server in each clique: \n")
    for key, value in cliques.items():
        print(f"{len(value)}", end=", ")
    print()'''


def pleroma_visibility(user_index, user_to_server, server_to_clique, cliques, servers):
    # Visibility(pleroma) = in same relay + direct follow
    
    server_id = user_to_server[user_index]
    clique_ids = server_to_clique[server_id]
    
    # get visible users
    all_visible_users = []
    for clique_id in clique_ids:
        for server in cliques[clique_id]:
            for j in servers[server]:
                if j not in all_visible_users:
                    all_visible_users.append(j)
    for j in servers[server_id]:
        if j not in all_visible_users:
            all_visible_users.append(j)
    all_visible_users.remove(user_index)
    
    return all_visible_users


def mastodon_visibility(user_index, user_to_server, server_to_clique, cliques, servers):
    server_id = user_to_server[user_index]
    clique_ids = server_to_clique[server_id]
    
    # get visible users
    all_visible_users = []
    for clique_id in clique_ids:
        for server in cliques[clique_id]:
            for j in servers[server]:
                if j not in all_visible_users:
                    all_visible_users.append(j)
    for j in servers[server_id]:
        if j not in all_visible_users:
            all_visible_users.append(j)
    all_visible_users.remove(user_index)
    
    return all_visible_users


# Initialize user-server graph and server-clique graph
def grow_network(n_nodes, user_to_server, server_to_clique, cliques, servers, visibility_func=pleroma_visibility):
    G = nx.DiGraph()
    H = nx.Graph()
    
    user_visible_universe = []# defaultdict(list)

    # Grow the user network
    for i in range(n_nodes):
        G.add_node(i)
        
    for i in range(n_nodes):
        # server_id = user_to_server[i]
        # clique_ids = server_to_clique[server_id]
        
        # # get visible users
        # all_visible_users = []
        # for clique_id in clique_ids:
        #     for server in cliques[clique_id]:
        #         for j in servers[server]:
        #             if j not in all_visible_users:
        #                 all_visible_users.append(j)
        # for j in servers[server_id]:
        #     if j not in all_visible_users:
        #         all_visible_users.append(j)
        # all_visible_users.remove(i)
        
        all_visible_users = pleroma_visibility(i, user_to_server, server_to_clique, cliques, servers)
        
        user_visible_universe.append(all_visible_users)

        # get number of followers for each visible user as weights for PA
        visible_users_followers = []
        for user in all_visible_users:
            num_followers = len(G.edges(user))
            visible_users_followers.append(num_followers)
        visible_users_followers = np.array(visible_users_followers) + 1 # to avoid user with no followers initially get no new followers
        
        # print(f"Node {i}")
        # print(len(all_visible_users))
        # print(all_visible_users)
        # print(f"Node {i} has {len(all_visible_users)} visible users")
        
        # denominator = sum(math.pow(i, 2) for i in lengths.values())
        # probs = [math.pow(n, 2)/denominator for n in lengths.values()] 
        
        denominator = sum(math.pow(i, 2) for i in visible_users_followers)#sum(visible_users_followers)
        try:
            m = int(power_law_samples(1, 1, 10, -2))
            if denominator/len(visible_users_followers) == 1:
                if len(all_visible_users) >= m:
                    targets = np.random.choice(all_visible_users, size=m)
                else:
                    targets = all_visible_users
                for j in range(len(targets)):
                    G.add_edge(i, targets[j])
                    # print(f"added edge from {i} to {targets[i]}")
                # print(f"Node {i} has {len(targets)} new followings with {len(all_visible_users)} visible users")
            else:
                probs = [math.pow(n, 2)/denominator for n in visible_users_followers] # visible_users_followers / denominator
                if len(all_visible_users) >= m:
                    targets = np.random.choice(all_visible_users, size=m, replace=False, p=probs)
                else:
                    targets = all_visible_users
                for j in range(len(targets)):
                    G.add_edge(i, targets[j])
                    # print(f"added edge from {i} to {targets[i]}")
                #print(f"Node {i} has {len(targets)} new followings with {len(all_visible_users)} visible users")
        except ValueError as e:
            print(e)
            print("Too few visible users that have followers to choose from. Adding by random.")
            if len(all_visible_users) >= m:
                targets = np.random.choice(all_visible_users, size=m)
            else:
                targets = all_visible_users
            for j in range(m):
                G.add_edge(i, targets[j])
                # print(f"added edge from {i} to {targets[i]}")
        except ZeroDivisionError as e:
            print(e)
            print(f"No visible users for node {i}")
                
    # Grow the server-clique network
    
                
    return G, np.array(user_visible_universe)


    

# for i in range(n_initial, n_initial + new_nodes):
#     G.add_node(i)
#     if np.random.rand() < new_server_probability:  # Form a new server
#         m = int(power_law_samples(1, 1, 5, -2.5))  # Number of initial nodes in the new server
#         new_server = max(servers.values()) + 1  # ID of the new server
#         for j in range(i, i+m):
#             G.add_node(j)
#             servers[j] = new_server
#             targets = np.random.choice(list(set(G.nodes()) - {j}), size=np.random.randint(1, 5), replace=False)
#             G.add_edges_from([(j, target) for target in targets])
#         i += m  # Skip the nodes that we just added

#         if np.random.rand() < new_clique_probability:  # Form a new clique
#             new_clique = max(server_cliques.values()) + 1  # ID of the new clique
#             server_cliques[new_server] = new_clique
#             H.add_node(new_server)
#         else:  # Join an existing clique
#             clique = np.random.choice(list(set(server_cliques.values())))  # Choose a clique to join
#             server_cliques[new_server] = clique
#             server_ids = [server for server, clique_id in server_cliques.items() if clique_id == clique]
#             for server_id in server_ids:
#                 if server_id != new_server:  # Avoid self-loops
#                     H.add_edge(new_server, server_id)
                    
#     else:  # Join an existing server
#         m = int(power_law_samples(1, 1, 5, -2.5))  # Number of edges to add
#         server_id = np.random.choice(list(set(servers.values())))  # Choose a server to join
#         servers[i] = server_id
#         server_nodes = [node for node, server in servers.items() if server == server_id and node != i]
#         other_nodes = [node for node, server in servers.items() if server != server_id and node != i]
#         for _ in range(m):
#             if np.random.rand() < in_server_edge_proba and server_nodes:  # Preferential attachment within the server
#                 targets = np.random.choice(server_nodes, size=1, replace=False)
#             elif other_nodes:  # Preferential attachment to other servers
#                 targets = np.random.choice(other_nodes, size=1, replace=False)
#             else:
#                 continue
#             G.add_edge(i, targets[0])


def prepare_show_graphs(G, servers, user_to_server):
    # Prepare the labels (in-degree of each node)
    labels = {node: G.in_degree(node) for node in G.nodes()}

    # Prepare the colors (based on server ID)
    colors = [user_to_server[node] for node in G.nodes()]

    # Group nodes by servers
    # nodes_by_server = defaultdict(list)
    # for server, node in servers.items():
    #     nodes_by_server[server].append(node)

    # Compute a separate spring layout for each server and adjust the positions
    pos = {}

    # Determine the number of servers per row/column in the grid
    servers_per_row = math.ceil(math.sqrt(len(servers)))

    x_shift = 0
    y_shift = 0
    for i, (server, nodes) in enumerate(servers.items()):
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, seed=42)
        # Adjust positions
        for k, v in pos_subgraph.items():
            v[0] += x_shift
            v[1] += y_shift
        pos.update(pos_subgraph)
        x_shift += 2.0  # Update the x-shift
        if (i + 1) % servers_per_row == 0:  # If this server is the last one in its row
            x_shift = 0  # Reset the x-shift
            y_shift -= 2.0  # Update the y-shift
            
    return pos, labels, colors

# Draw the graph
def show_onebyone(G, pos, labels, colors, H, server_cliques):
    plt.figure(figsize=(16,10))
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.get_cmap('rainbow'))
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title('Directed Network with node in-degrees')
    plt.show()

    # Draw the server-clique graph
    color_map = []
    for node in H:
        color_map.append(server_cliques[node])

    nx.draw(H, node_color=color_map, with_labels=True, cmap=plt.get_cmap('rainbow'))
    plt.title('Server-clique network')
    plt.show()

def show_together(G, pos, labels, colors, H, server_cliques):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Increased figure size and created 2 subplots

    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.get_cmap('rainbow'), ax=axs[0])
    nx.draw_networkx_labels(G, pos, labels=labels, ax=axs[0])
    axs[0].set_title('Directed Network with node in-degrees')

    color_map = []
    for node in H:
        color_map.append(server_cliques[node])

    nx.draw(H, node_color=color_map, with_labels=True, cmap=plt.get_cmap('rainbow'), ax=axs[1])
    axs[1].set_title('Server-clique network')

    # Adjust the subplot layout
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.4)

    plt.show()


def power_law_samples(n, min_value, max_value, exponent):
    r = np.random.uniform(0, 1, n)
    return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))

if __name__ == '__main__':
    main()