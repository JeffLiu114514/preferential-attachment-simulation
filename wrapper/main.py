from generation import GenerationWrapper

if __name__ == "__main__":
    n_nodes = 10
    user_server_ratio = 0.2
    server_clique_ratio = 1
    clique_size_expectation = 5
    n_servers = int(n_nodes * user_server_ratio)
    n_cliques = int(n_servers * server_clique_ratio)
    posts_hyperparameters = [1, 0.6, 0.2, 0.2, 0.4, 0.6, 0, 0.2, 0.1, 0.1] # alpha1, alpha2, alpha3, beta1, beta2, beta3, engagement_threshold, follow2, follow3, unfollow1
    
    gw = GenerationWrapper(n_nodes, n_servers, n_cliques, clique_size_expectation, posts_hyperparameters)
    # print(gw.user_visible_by_follow)
    # gw.visualize_user_graph_networkx()
    gw.visualize_user_graph_graphviz()
    # for i in range(5):
    gw.update_network(5)
    for post in gw.posts_log:
        for positive in post.positive_engagements:
            print(positive)
        for negative in post.negative_engagements:
            print(negative)
    gw.visualize_user_graph_graphviz(filename="after_posts.gv")