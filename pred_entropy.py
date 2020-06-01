# -*-coding:utf-8-*-
'''
@Time: 2020/1/9 15:17
@Author: xshura
@File: pred_entropy.py
@Mail: 2209032305@qq.com
'''
import math

from sklearn import ensemble
import numpy as np
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)


def entropy(c):
    classes = {}
    for x in c:
        if x not in classes.keys():
            classes[x] = 1
        else:
            classes[x] += 1
    e = 0.0
    for k in classes:
        p = classes[k] / len(c)
        e += (-p) * math.log(p, 2)
    return e


def get_node_entropy(g, labels):
    labels = labels.tolist()
    node_edges = {}
    for k in g.nodes:
        temp = [labels[x].index(1) for x in g[k]]
        node_edges[k] = temp

    for k, v in node_edges.items():
        node_edges[k] = entropy(v)
    nodes_entropy = sorted(node_edges.items(), key=lambda x: x[1])
    res = {}
    for x, y in nodes_entropy:
        res[x] = y
    return res


def pred_entropy(nodes, edges, degrees, ally, ty):
    labels = np.vstack([ally, ty])
    node_entropy = get_node_entropy(edges, labels)
    x_degree = [degrees[x] for x in range(len(ally))]
