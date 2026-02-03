import matplotlib.pyplot as plt
import networkx as nx

pos = {n: (n[1], n[0]) for n in G.nodes}
nx.draw(G, pos, node_color='red', edge_color='blue', with_labels=True)
plt.imshow(skeleton, cmap='gray', alpha=0.5)
plt.show()
