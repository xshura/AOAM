'''
@Time: 2019/12/20 14:08
@Author: xshura
@File: environment.py
@Mail: 2209032305@qq.com
'''
import numpy as np
# from train import rating
from train import rating
import pickle as pkl
import sys
import networkx as nx
from load_data import compute_num_edges_nodes


def load(dataset_str):
    DATA_PATH = 'data/all_data/'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    g = nx.from_dict_of_lists(graph)
    edges = np.array(list(g.edges()))
    node_degrees = compute_num_edges_nodes(edges)
    return node_degrees, edges


class Env:
    def __init__(self, data, top_num, dataset):
        self.dataset = dataset
        self.top_num = top_num
        self.data = data
        self.cur_state = data.copy()
        self.actions = []
        self.iter = 0
        self.nodes_degree, self.edges = load(dataset)
        nodes = list(self.nodes_degree.keys())
        self.top_nodes = nodes[:top_num]
        self.degrees = [0, 78, 67, 35, 31, 29]
        self.done = False

    def step(self, action):
        reward = 0
        self.cur_state[self.iter][0] = action[0]
        self.actions.append(action[0])
        self.iter += 1
        observation_ = np.reshape(self.cur_state, [len(self.cur_state)])
        if self.iter >= self.top_num:
            self.done = True
            print(self.actions)
            reward = rating(observation_, self.dataset)
        return self.cur_state, reward, self.done, "useless"

    def reset(self):
        self.cur_state =self.data.copy()
        self.done = False
        self.iter = 0
        self.actions = []
        return self.data.copy()

    def render(self):
        print("I am rendering!!!! ")

