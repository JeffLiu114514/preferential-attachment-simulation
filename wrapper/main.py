from generation import GenerationWrapper
from collections import defaultdict
import numpy as np
import scipy
import math

import itertools

def kendallTau(A, B):
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            distance += 1

    return distance


if __name__ == "__main__":
    n_nodes = 500
    user_server_ratio = 0.2
    server_clique_ratio = 1
    clique_size_expectation = 5
    n_servers = int(n_nodes * user_server_ratio)
    n_cliques = int(n_servers * server_clique_ratio)
    
    posts_hyperparameters = [1, 0.6, 0.2, 0.2, 0.4, 0.6, 0, 0.2, 0.1, 0.1] 
    # alpha1, alpha2, alpha3, beta1, beta2, beta3, engagement_threshold, follow2, follow3, unfollow1
    
    gw = GenerationWrapper(n_nodes, n_servers, n_cliques, clique_size_expectation, posts_hyperparameters)
    
    true_scores = defaultdict(list)
    discrete_scores = defaultdict(list)
    for i in range(50):
        post = gw.make_post(i)
        scores = gw.post_reactions(post)
        for user, score in scores.items():
            
            
            discrete_scores[user].append(score[0])
            true_scores[user].append(score[1])
    
    users_to_search = list(set(discrete_scores.keys()).union(set(true_scores.keys())))
    for i in users_to_search:
        discrete_score = discrete_scores[i]
        true_score = true_scores[i]
        if len(discrete_score) == 1 or len(true_score) == 1:
            continue
        discrete_ranking = np.argsort(discrete_score)
        true_ranking = np.argsort(true_score)
        kendall_tau_correlation = scipy.stats.kendalltau(discrete_ranking, true_ranking)
        kendall_tau_distance = kendallTau(discrete_score, true_score)
        # print(discrete_ranking.size)
        # print(len(discrete_ranking))
        kendall_tau_distance_normalized = kendall_tau_distance / math.comb(discrete_ranking.size, 2)
        print(discrete_score, true_score)
        print(discrete_ranking, true_ranking)
        print("Kendall Tau Correlation: ", kendall_tau_correlation)
        print("Kendall Tau Distance: ", kendall_tau_distance)
        print("Kendall Tau Distance Normalized: ", kendall_tau_distance_normalized)
        
    
    # print(gw.user_visible_by_follow)
    # gw.visualize_user_graph_networkx()
    # gw.visualize_user_graph_graphviz()
    # # for i in range(5):
    # gw.update_network(5)
    # for post in gw.posts_log:
    #     for positive in post.positive_engagements:
    #         print(positive)
    #     for negative in post.negative_engagements:
    #         print(negative)
    # gw.visualize_user_graph_graphviz(filename="after_posts.gv")