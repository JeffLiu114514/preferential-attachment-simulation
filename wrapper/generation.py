import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
import math
from graphviz import Digraph
from utils import *

import os
os.environ["PATH"] += os.pathsep + '../Graphviz/bin'

class User:
    def __init__(self, index, followers, followings, server_visibility, relay_visibility, mastodon_visibility=None, post_quality_distribution=(0, 1)):
        self.index = index
        self.post_quality_distribution = post_quality_distribution
        self.followers = followers
        self.followings = followings
        self.server_visibility = server_visibility
        self.relay_visibility = relay_visibility
        self.mastodon_visibility = mastodon_visibility
        self.posts = []
        self.update_visibility()
        
    def update_visibility(self):
        self.first_visibility = set(self.followers) # those who can see my posts
        self.second_visibility = set(self.server_visibility) - self.first_visibility
        self.third_visibility = set(self.relay_visibility) - self.first_visibility - self.second_visibility
        if self.mastodon_visibility is not None:
            self.mastodon_visibility = set(self.mastodon_visibility) - self.first_visibility


class Post:
    def __init__(self, index, user_index, quality):
        self.index = index
        self.user_index = user_index
        self.quality = quality
        self.positive_engagements = []
        self.negative_engagements = []
    

class GenerationWrapper:
    
    def __init__(self, n_nodes, n_servers, n_cliques, clique_size_expectation, post_hyperparams, user_quality_distribution=(0, 1)) -> None:
        self.n_nodes = n_nodes
        self.n_servers = n_servers
        self.n_cliques = n_cliques
        self.post_hyperparams = post_hyperparams
        self.user_quality_distribution = user_quality_distribution
        self.generate_initial_network(clique_size_expectation)
        
        
    def update_network(self, n_posts):
        self.posts_log = []
        for i in range(n_posts):
            post = self.make_post(i)
            self.posts_log.append(post)
        
    def make_post(self, post_counter):
        '''
        func making_post():
            sample one User from user distribution weighted by # of followers
            User makes Post by sampling quality from quality distribution
            for user in User's visibility: (repeat for 1st 2nd 3rd visibility)
                if quality > 0: # positive engagement
                    determine engagement parameter by quality and alpha
                    if engagement parameter > engagement threshold:
                        positive engagement
                        determine whether to follow (if in 2nd or 3rd visibility)
                else: # negative engagement
                    determine engagement parameter by quality and beta
                    if engagement parameter < -engagement threshold:
                        negative engagement
                        determine whether to unfollow (if in 1st visibility)
                
                ---------------(we don't consider whether more engagement makes a post more possible to get engaged for other users later)----------------
        '''
        user = linear_PA_probs2(self.users)
        quality = np.random.normal(user.post_quality_distribution[0], user.post_quality_distribution[1])
        post = Post(post_counter, user.index, quality)
        print(f"Post {post_counter} made by user {user.index} with quality {quality}")
        alpha1, alpha2, alpha3, beta1, beta2, beta3, engagement_threshold, follow2, follow3, unfollow1 = self.post_hyperparams
        
        for other_user in user.first_visibility:
            observed_quality = np.random.normal(quality, alpha1)
            if quality >= 0:
                
                if engagement_param > engagement_threshold:
                    post.positive_engagements.append({"poster": user.index, "engager": other_user, "visibility": "1st", "type":"positive", "status": "unchanged"})
            else:
                engagement_param = beta1 * quality
                if engagement_param < -engagement_threshold:
                    if random01(unfollow1):
                        user.followers.remove(other_user)
                        self.users[other_user].followings.remove(user.index)
                        self.G.remove_edge(other_user, user.index)
                        post.negative_engagements.append({"poster": user.index, "engager": other_user, "visibility": "1st", "type":"negative", "status": "unfollowed"})
                    else:
                        post.negative_engagements.append({"poster": user.index, "engager": other_user, "visibility": "1st", "type":"negative", "status": "unchanged"})
        
        for other_user in user.second_visibility:
            if quality >= 0:
                engagement_param = alpha2 * quality
                if engagement_param > engagement_threshold:
                    if random01(follow2):
                        user.followers.append(other_user)
                        self.users[other_user].followings.append(user.index)
                        self.G.add_edge(other_user, user.index)
                        post.positive_engagements.append({"poster": user.index, "engager": other_user, "visibility": "2nd", "type":"positive", "status": "followed"})
                    else:
                        post.positive_engagements.append({"poster": user.index, "engager": other_user, "visibility": "2nd", "type":"positive", "status": "unchanged"})
            else:
                engagement_param = beta2 * quality
                if engagement_param < -engagement_threshold:
                    post.negative_engagements.append({"poster": user.index, "engager": other_user, "visibility": "2nd", "type":"negative", "status": "unchanged"})
                    
        for other_user in user.third_visibility:
            if quality >= 0:
                engagement_param = alpha3 * quality
                if engagement_param > engagement_threshold:
                    if random01(follow3):
                        user.followers.append(other_user)
                        self.users[other_user].followings.append(user.index)
                        self.G.add_edge(other_user, user.index)
                        post.positive_engagements.append({"poster": user.index, "engager": other_user, "visibility": "3rd", "type":"positive", "status": "followed"})
                    else:
                        post.positive_engagements.append({"poster": user.index, "engager": other_user, "visibility": "3rd", "type":"positive", "status": "unchanged"})
            else:
                engagement_param = beta3 * quality
                if engagement_param < -engagement_threshold:
                    post.negative_engagements.append({"poster": user.index, "engager": other_user, "visibility": "3rd", "type":"negative", "status": "unchanged"})
                    
        return post
                        

    def generate_initial_network(self, clique_size_expectation):
        self.servers, self.user_to_server = self.generate_user_server_mapping()
        self.cliques, self.server_to_clique = self.generate_server_clique_mapping_two(clique_size_expectation)
        self.G, self.user_visible_by_relay = self.grow_network()
        self.user_visible_by_server = self.server_visibility()
        self.user_visible_by_follow = self.find_direct_followings()
        
        self.users = []
        for i in range(self.n_nodes):
            user_mean_quality = np.random.normal(self.user_quality_distribution[0], self.user_quality_distribution[1])
            user = User(i, self.find_direct_followers()[i], self.user_visible_by_follow[i], self.user_visible_by_server[i], self.user_visible_by_relay[i], 
                        post_quality_distribution=(user_mean_quality, 1))
            self.users.append(user)
        
    
    def generate_user_server_mapping(self):
        # each server has at least one user
        servers = defaultdict(list)
        for i in range(self.n_servers):
            servers[i].append(i)

        # add users to servers by PA only depending on the number of users in the server
        for i in range(self.n_servers, self.n_nodes):
            probs = linear_PA_probs(servers)
            
            server_to_join = np.random.choice(range(self.n_servers), size=1, replace=False, p=probs)
            servers[server_to_join[0]].append(i)

        user_to_server = {}
        # get reverse mapping
        for server_id, user_ids in servers.items():
            for user_id in user_ids:
                user_to_server[user_id] = server_id
                
        return servers, user_to_server


    # second approach: flipping coins
    def generate_server_clique_mapping_two(self, clique_size_expectation):
        cliques = defaultdict(set)
        server_to_clique = defaultdict(set)

        server_ids = list(self.servers.keys())
        server_weights = np.array([len(self.servers[id]) for id in server_ids]).astype(float)
        server_weights /= np.sum(server_weights)

        for clique in range (self.n_cliques):
            for server in server_ids:
                if np.random.rand() < (server_weights[server] * clique_size_expectation):
                    cliques[clique].add(server)
                    server_to_clique[server].add(clique)
            
            #print(f"{clique}th clique")
            
        return cliques, server_to_clique
    
    
    def grow_network(self):
        G = nx.DiGraph()

        # Grow the user network
        for i in range(self.n_nodes):
            G.add_node(i)
            
        user_visible_universe = self.relay_visibility()
        for i in range(self.n_nodes):
            all_visible_users = user_visible_universe[i]

            # get number of followers for each visible user as weights for PA
            visible_users_followers = []
            for user in all_visible_users:
                num_followers = len(G.edges(user))
                visible_users_followers.append(num_followers)
            visible_users_followers = np.array(visible_users_followers) + 1 # to avoid user with no followers initially get no new followers
            
            # apply PA to get new followings
            denominator = sum(math.pow(i, 2) for i in visible_users_followers)# quadratic PA               #sum(visible_users_followers)
            try:
                m = int(power_law_samples(1, 1, 10, -2))# number of new followings from user i
                if denominator == len(visible_users_followers): # all visible users have no followers
                    if len(all_visible_users) >= m:
                        targets = np.random.choice(all_visible_users, size=m)
                    else:
                        targets = all_visible_users
                    for j in range(len(targets)):
                        G.add_edge(i, targets[j])
                else:
                    probs = [math.pow(n, 2)/denominator for n in visible_users_followers] # visible_users_followers / denominator
                    if len(all_visible_users) >= m:
                        targets = np.random.choice(all_visible_users, size=m, replace=False, p=probs)
                    else:
                        targets = all_visible_users
                    for j in range(len(targets)):
                        G.add_edge(i, targets[j])
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
                    
        return G, user_visible_universe
    
    
    def relay_visibility(self):
        # Visibility = in same relay
        user_visible_universe = []
        
        for user_index in range(self.n_nodes):
            server_id = self.user_to_server[user_index]
            clique_ids = self.server_to_clique[server_id]
            
            # get visible users
            all_visible_users = []
            for clique_id in clique_ids:
                for server in self.cliques[clique_id]:
                    for j in self.servers[server]:
                        if j not in all_visible_users:
                            all_visible_users.append(j)
            for j in self.servers[server_id]:
                if j not in all_visible_users:
                    all_visible_users.append(j)
            all_visible_users.remove(user_index)
            user_visible_universe.append(all_visible_users)
        
        return user_visible_universe
    
    def server_visibility(self):
        user_visible_by_server = []
        for user_index in range(self.n_nodes):
            all_visible_users = []
            server_id = self.user_to_server[user_index]
            for user in self.servers[server_id]:
                if user not in all_visible_users:
                    all_visible_users.append(user)
            all_visible_users.remove(user_index)
            user_visible_by_server.append(all_visible_users)

        return user_visible_by_server
        
    def mastodon_visibility(self):
        # Visibility = all people in the same server's direct followings
        user_visible_by_server_mastodon = []
        for user_index in range(self.n_nodes):
            peoples_follow = []
            users_in_same_server = self.user_visible_by_server[user_index]
            for user in users_in_same_server:
                peoples_follow += self.user_visible_by_follow[user]
            peoples_follow.remove(user_index)
            user_visible_by_server_mastodon.append(set(peoples_follow))
        
        return user_visible_by_server_mastodon
            
    
    def find_direct_followers(self):
        return [[edge[0] for edge in self.G.in_edges(i)] for i in range(self.n_nodes)]
    
    def find_direct_followings(self):
        return [[edge[1] for edge in self.G.out_edges(i)] for i in range(self.n_nodes)]
    
    def visualize_user_graph_networkx(self):
        labels = {node: self.G.in_degree(node) for node in self.G.nodes()}

        # Prepare the colors (based on cluster ID)
        colors = [self.user_to_server[node] for node in self.G.nodes()]

        # Compute a separate spring layout for each cluster and adjust the positions
        pos = {}

        # Determine the number of clusters per row/column in the grid
        servers_per_row = math.ceil(math.sqrt(len(self.servers)))

        x_shift = 0
        y_shift = 0
        for i, (cluster, nodes) in enumerate(self.servers.items()):
            subgraph = self.G.subgraph(nodes)
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
        nx.draw(self.G, pos, node_color=colors, with_labels=False, node_size=200, width=0.5, edge_vmax=0.5, cmap=plt.get_cmap('rainbow'))
        nx.draw_networkx_labels(self.G, pos, labels=labels)
        plt.title('Directed Network with node in-degrees')
        plt.show()
        
        
    def visualize_user_graph_graphviz(self, filename='clusters.gv'):
        g = Digraph('G', filename=filename)
        for i in range(self.n_servers):
            with g.subgraph(name=f'cluster_{i}') as c:
                c.attr(style='filled', color='lightgrey')
                c.node_attr.update(style='filled', color='white')
                for user in self.servers[i]:
                    c.node(str(user))
                c.attr(label=f'server #{i}')
        
        for edge in self.G.edges():
            g.edge(str(edge[0]), str(edge[1]))
            
        g.view()