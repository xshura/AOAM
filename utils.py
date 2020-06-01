from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import matplotlib.pyplot as plt
import networkx as nx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def gauss(arange):
    # 高斯
    import numpy as np
    import math
    u = 1  # 均值μ
    sig = math.sqrt(0.2)  # 标准差δ
    x = np.linspace(u - 3 * sig, u + 3 * sig, arange)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    return y_sig

def compute_num_edges_nodes(edges):
    node_edges = {}
    for e in edges:
        if e[0] in node_edges:
            node_edges[e[0]] = node_edges[e[0]] + 1
        else:
            node_edges[e[0]] = 1
        # if e[1] in node_edges:
        #     node_edges[e[0]] = node_edges[e[0]] + 1
        # else:
        #     node_edges[e[0]] = 1
    degrees = sorted(node_edges.items(), key=lambda x: x[1])[::-1]
    res = {}
    for x, y in degrees:
        res[x] = y
    return res

def get_node_degree(path="data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    node_degree = compute_num_edges_nodes(edges)
    return node_degree, edges


def load_data(path="data/cora/", dataset="cora", score=None):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    def show_graph(edges):
        plt.figure(figsize=(20, 20))
        g = nx.Graph()
        for i in range(0, 100):
            g.add_weighted_edges_from([(edges[i][0], edges[i][1], 1)])
        pos = nx.random_layout(g)
        nx.draw_networkx_edges(g, pos, with_labels=True, edge_color='black', alpha=0.8, font_size=10,
                               width=2)
        nx.draw_networkx_nodes(g, pos, with_labels=True)
        nx.draw_networkx_labels(g, pos)
        plt.show()
    if dataset == 'cora':
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        # for i in range(len(idx_features_labels)):
        #     if idx_features_labels[i, 0] == 'abberley99thisl':
        #         print("wtf", i)
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        # sort node by degree
        node_degree = compute_num_edges_nodes(edges)
        nodes = list(node_degree.keys())[:5]
        score = np.ones([len(edges)])
        # mask = np.array([x for x in range(0, score.shape[0])])
        # np.random.shuffle(mask)
        # mask = mask.tolist()[:300]
        # top_edges = []
        for i, e in enumerate(edges.tolist()):
            # if i in mask:
            #     score[i] = 0
            for v in nodes:
                if v in e:
                    score[i] = 3

        # show_graph(top_edges)
        if type(score) == np.ndarray:
            adj = sp.coo_matrix((score, (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        else:
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
        return features.todense(), adj, labels
    elif dataset == 'citeseer':

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        g = nx.read_edgelist("{}{}.cites".format(path, dataset),delimiter='\t', create_using=nx.DiGraph())
        edges = list(nx.edges(g))
        score = np.ones([len(edges)])
        edges = [(edges[i][0], edges[i][1], score[i]) for i in range(len(edges))]
        g.add_weighted_edges_from(edges)
        adj = nx.adjacency_matrix(g)
        # Transpose the adjacency matrix, as Citeseer raw dataset comes with a
        # <ID of cited paper> <ID of citing paper> edgelist format.
        adj = adj.T
        return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1000)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()
    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two
    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape