import networkx as nx

from node_embeddings_clustering import graph_clustering

CLUSTERS_PROBS_TEST_3 = [[0.75, 0.015, 0.0002], [0.015, 0.85, 0.0075], [0.0002, 0.0075, 0.90]]

CLUSTERS_SIZES_TEST_3 = [20, 50, 30]

graph = nx.stochastic_block_model(CLUSTERS_SIZES_TEST_3, CLUSTERS_PROBS_TEST_3, seed=0)

clusters = graph_clustering(graph)

print(clusters)