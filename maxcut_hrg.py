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
import cvxpy as cp
from scipy.linalg import sqrtm


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
    n = len(G)
    mapping = dict(zip(G, range(0, n)))
    G = nx.relabel_nodes(G, mapping)
    node_positions = {};
    for v in G.nodes():
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
    nx.draw_networkx(G, pos=node_positions, nodelist=list(cut[1]), node_color = 'red', with_labels = False);
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

def gw_max_cut(G, weight=None):
    #relabel vertices
    n = len(G)
    mapping = dict(zip(G, range(0, n)))
    G = nx.relabel_nodes(G, mapping)

    #Declare a positive semidefinite matrix
    A = cp.Variable((n, n), symmetric=True)

    #Set diagonals 1
    constraints = [A >> 0]
    constraints += [
        A[i, i] == 1 for i in range(n)
    ]

    #Set objective function accordingly to our relaxed version of the problem
    edges = [e for e in G.edges]
    objective = sum(0.5*(1-A[i,j]) for (i, j) in edges)

    #solve positive semedefinite Matrix
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    #get the vectors with norm ||x||_2 = 1
    y = sqrtm(A.value)
    #generate a random hyperplane
    r = np.random.randn(n)
    #pick labels according to the side of the hyperplane
    y = np.sign(y @ r)

    cut = {node for node in G.nodes() if y[node] > 0}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return cut_size, partition



n_Graph = [250, 500, 750, 1000]
for g_n in n_Graph:
    G, G_all, pos = HRG_generator(g_n, 0.75, 0, 10)
    print("Giant Component size: ", len(G))
    size_greedy, cut_greedy = greedy_max_cut(G)
    print("Greedy Cut: ", size_greedy, nx.number_of_edges(G))
    drawCut(G, pos, cut_greedy)

    size_gw, cut_gw = gw_max_cut(G)
    print("GW Cut: ", size_gw, nx.number_of_edges(G))
    drawCut(G, pos, cut_gw)
