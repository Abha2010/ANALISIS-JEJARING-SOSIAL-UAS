import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Membaca dataset edge list
df = pd.read_csv("../dataset/facebook_combined_1000.txt",
                 sep=" ", header=None, names=["source", "target"])

G = nx.from_pandas_edgelist(df, "source", "target")

print("Jumlah Node :", G.number_of_nodes())
print("Jumlah Edge :", G.number_of_edges())

# Centrality
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=2000)

print("\nNode Degree tertinggi:",
      max(degree, key=degree.get), max(degree.values()))
print("Node Betweenness tertinggi:",
      max(betweenness, key=betweenness.get), max(betweenness.values()))
print("Node Closeness tertinggi:",
      max(closeness, key=closeness.get), max(closeness.values()))
print("Node Eigenvector tertinggi:",
      max(eigenvector, key=eigenvector.get), max(eigenvector.values()))

# Global metrics
print("\nDensity :", nx.density(G))
print("Diameter :", nx.diameter(G))
print("Average Path Length :", nx.average_shortest_path_length(G))
print("Clustering Coefficient :", nx.average_clustering(G))

# Louvain
communities = nx.community.louvain_communities(G, seed=42)
print("Jumlah komunitas Louvain :", len(communities))

# Visualisasi
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 9))
nx.draw(G, pos, node_size=8, edge_color="gray", alpha=0.4, with_labels=False)
plt.title("Jejaring Sosial 1.000 Node")
plt.show()
