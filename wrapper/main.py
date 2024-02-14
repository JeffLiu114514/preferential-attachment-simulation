from generation import GenerationWrapper

if __name__ == "__main__":
    n_nodes = 20
    user_server_ratio = 0.1
    server_clique_ratio = 1
    clique_size_expectation = 5
    n_servers = int(n_nodes * user_server_ratio)
    n_cliques = int(n_servers * server_clique_ratio)
    
    gw = GenerationWrapper(n_nodes, n_servers, n_cliques, clique_size_expectation)
    print(gw.user_visible_by_relay)
    # print(gw.user_visible_by_follow)
    # gw.visualize_user_graph_networkx()
    gw.visualize_user_graph_graphviz()