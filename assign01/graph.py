import networkx as nx
import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True

A = "$A$"
B = "$B$"
C = "$C$"
D = "$D$"

D_A = "$D_A = \set{m, p}$"
D_B = "$D_B = \set{t, f, b}$"
D_C = "$D_C = \set{st, r, h}$"
D_D = "$D_D = \set{ph, s, d}$"

G = nx.Graph()

nodeA = f'{A}\n{D_A}'
nodeB = f'{B}\n{D_B}'
nodeC = f'{C}\n{D_C}'
nodeD = f'{D}\n{D_D}'

G.add_nodes_from([nodeA, nodeB, nodeC, nodeD])

edgeAC = nodeA, nodeC
edgeAB = nodeA, nodeB
edgeBD = nodeB, nodeD
edge_msgs = {
    edgeAC: "$A = m | C \in \set{st, h}$",
    edgeAB: "$A = p \Rightarrow B = t$",
    edgeBD: "$B = b | D \in \set{s, d}$"
}

G.add_edges_from([(nodeA, nodeB), (nodeA, nodeC), (nodeB, nodeD), (nodeC, nodeD)])

pos = nx.spring_layout(G)

nx.draw_networkx(G, with_labels=True, node_color='lightgreen', node_size=4000, 
                 font_size=10, font_weight='bold', edge_color='gray', width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_msgs, font_color='red', font_size=9)

plt.title("Example Node Graph")
plt.axis("off")
plt.show()