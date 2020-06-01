from keras.models import load_model
from layers.graph import GraphConvolution
from utils import *
from load_data import load_data_by_str


def rating(weights, dataset, alpha=None):
    FILTER = 'localpool'  # 'chebyshev'
    MAX_DEGREE = 2  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    A, X, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = load_data_by_str(dataset, score=weights, alpha=alpha)
    if dataset != 'citeseer':
        X /= X.sum(1).reshape(-1, 1)
    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        # print('Using local pooling filters...')
        A_ = preprocess_adj(A, SYM_NORM)
        graph = [X, A_]
    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        support = MAX_DEGREE + 1
        graph = [X]+T_k
    else:
        raise Exception('Invalid filter type.')
    model = load_model('{}_model.h5'.format(dataset), custom_objects={"GraphConvolution": GraphConvolution})
    preds = model.predict(graph, batch_size=A.shape[0])

    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    preds_txt = np.argmax(preds[[idx_test]], 1)
    return test_acc[0], preds_txt


if __name__ == '__main__':
    dataset = "cora"
    # load best adj score
    score = np.loadtxt("best_model/{}_best_score.txt".format(dataset))
    origin_acc, _ = rating(None, dataset)
    aoam_acc, _ = rating(score, dataset)
    print("Dataset: {}".format(dataset))
    print("origin accuracy is:", origin_acc)
    print("AOAM accuracy is:", aoam_acc)
