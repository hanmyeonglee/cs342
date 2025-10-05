import networkx as nx
import matplotlib.pyplot as plt
import os
import math
import random

current_dir = os.path.dirname(os.path.abspath(__file__))

#plt.rcParams['text.usetex'] = True

A = "$A$"
B = "$B$"
C = "$C$"
D = "$D$"

D_A = "$D_A = set{m, p}$"
D_B = "$D_B = set{t, f, b}$"
D_C = "$D_C = set{st, r, h}$"
D_D = "$D_D = set{ph, s, d}$"

G = nx.Graph()

nodeA = f'{A}\n{D_A}'
nodeB = f'{B}\n{D_B}'
nodeC = f'{C}\n{D_C}'
nodeD = f'{D}\n{D_D}'

G.add_nodes_from([nodeA, nodeB, nodeC, nodeD])

edgeAC = nodeA, nodeC
edgeAB = nodeB, nodeA
edgeBD = nodeB, nodeD
edge_msgs = {
    edgeAC: "$A = m | C in set{st, h}$",
    edgeAB: "$A = p Rightarrow B = t$",
    edgeBD: "$B = b | D in set{s, d}$"
}

G.add_edges_from([edgeAC, edgeAB, edgeBD])

angle_offset = math.pi / 4
radius = 1.25
random_offset = 0.2

pos = {
    nodeA: (
        radius * math.cos(0 + angle_offset) + random.uniform(-random_offset, random_offset),
        radius * math.sin(0 + angle_offset) + random.uniform(-random_offset, random_offset)
    ),
    nodeB: (
        radius * math.cos(math.pi + angle_offset) + random.uniform(-random_offset, random_offset),
        radius * math.sin(math.pi + angle_offset) + random.uniform(-random_offset, random_offset)
    ),
    nodeC: (
        radius * math.cos(math.pi / 2 + angle_offset) + random.uniform(-random_offset, random_offset),
        radius * math.sin(math.pi / 2 + angle_offset) + random.uniform(-random_offset, random_offset)
    ),
    nodeD: (
        radius * math.cos(3 * math.pi / 2 + angle_offset) + random.uniform(-random_offset, random_offset),
        radius * math.sin(3 * math.pi / 2 + angle_offset) + random.uniform(-random_offset, random_offset)
    ),
}

nx.draw_networkx_nodes(
    G, pos, node_color='white', node_size=4000, 
    linewidths=2, edgecolors='black'
)

nx.draw_networkx_edges(
    G, pos, edge_color='black', width=2
)

nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_msgs, font_color='black', font_size=10,
    horizontalalignment='center'
)

nx.draw_networkx_labels(
    G, pos, font_size=10, font_color='black'
)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.title("Constraint Graph over the A, B, C, and D")
plt.axis("off")

#plt.show()
plt.savefig(os.path.join(current_dir, "report_img", "constraint_graph.png"), bbox_inches='tight')
