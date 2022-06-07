import networkit as nk
import networkx as nx
from pygirgs import hypergirgs
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import copy
import math


def HRG_generator(n, alpha, t, deg):
    rseed = random.randrange(10000);
    aseed = random.randrange(10000);
    sseed = random.randrange(10000);

    R = hypergirgs.calculate_radius(n, alpha, t, deg);
    radii = hypergirgs.sample_radii(n, alpha, R, rseed, False);
    angles = hypergirgs.sample_angles(n, aseed, False);
    edges = hypergirgs.generate_edges(radii, angles, t, R, sseed);
    # "All" means here all connected components
    G_all = nx.Graph();
    for u, v in edges:
        G_all.add_edge(u,v);
    connected_components = list(nx.connected_components(G_all));
    max_nodes = 0;
    indexCC = -1;
    for i,nodesTemp in enumerate(connected_components):
        size = len(nodesTemp);
        if(max_nodes < size):
            max_nodes = size;
            indexCC = i;
    G_temp = G_all.subgraph(list(connected_components[indexCC]));
    G = nx.Graph();
    for u,v in G_temp.edges():
        G.add_edge(u,v);
    node_positions = {};
    for v in G_all.nodes():
        node_positions[v] = (radii[v]*math.cos(angles[v]),radii[v]*math.sin(angles[v]));
    return G, G_all,node_positions;

def drawGraph(G, nodePos, showNode=[], showNodeSets = None, title = ""):
    node_positions = nodePos;
    nx.draw_networkx(G, pos=node_positions, node_color = 'yellow', with_labels = False);

    cmap = plt.cm.get_cmap('hsv');
    if(showNodeSets != None):
        size = len(showNodeSets);
        colors = {};
        dist = 1./size;
        for i in range(size):
            colors[i] = np.array([cmap(i*dist)]);
        for i in range(size):
            nx.draw_networkx_nodes(G, pos=node_positions, nodelist = showNodeSets[i], node_color=colors[i]);
    else:
        nx.draw_networkx_nodes(G, pos=node_positions, nodelist = showNode, node_color='b');

    plt.title(title);
    plt.show();

def drawCut(G, nodePos, cut, showNode=[], showNodeSets = None, title = ""):
    node_positions = nodePos;
    nx.draw_networkx(G, pos=node_positions, nodelist=list(cut[0]), node_color = 'yellow', with_labels = False);
    nx.draw_networkx(G, pos=node_positions, nodelist=list(cut[1]), node_color = 'black', with_labels = False);
    plt.title(title);
    plt.show();


def randomized_max_cut(G, seed=None, p=0.5, weight=None):
    cut = {node for node in G.nodes() if random.uniform(0, 1) < p}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return cut_size, partition

def _swap_node_partition(cut, node):
    return cut - {node} if node in cut else cut.union({node})

def greedy_max_cut(G, initial_cut=None, seed=None, weight=None):
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    while True:
        nodes = list(G.nodes())
        # Shuffling the nodes ensures random tie-breaks in the following call to max
        random.shuffle(nodes)
        best_node_to_swap = max(
            nodes,
            key=lambda v: nx.algorithms.cut_size(
                G, _swap_node_partition(cut, v), weight=weight
            ),
            default=None,
        )
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)

        if potential_cut_size > current_cut_size:
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break

    partition = (cut, G.nodes - cut)
    return current_cut_size, partition

n_Graph = [250, 500, 750, 1000]
for g_n in n_Graph:
    G, G_all, pos = HRG_generator(g_n, 0.75, 0, 10)
    print(len(G), len(G_all))
    size, cut = greedy_max_cut(G)
    print(size, nx.number_of_edges(G))
    drawCut(G, pos, cut)
