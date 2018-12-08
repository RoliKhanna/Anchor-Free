
from nltk.corpus import reuters
import sys
import numpy as np
from scipy import optimize

# Loading data here
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

def SPA(X, r):
    cols = []
    m, n = X.shape
    assert(m == n)
    for _ in xrange(r):
        col_norms = np.sum(np.abs(X) ** 2, axis=0)
        col_ind = np.argmax(col_norms)
        cols.append(col_ind)
        col = np.reshape(X[:, col_ind], (n, 1))
        X = np.dot((np.eye(n) - np.dot(col, col.T) / col_norms[col_ind]), X)
    return cols

def col2norm(X):
    return np.sum(np.abs(X) ** 2,axis=0)

def NNLSFrob(X, cols):

    ncols = X.shape[1]
    H = np.zeros((len(cols), ncols))
    for i in xrange(ncols):
        sol, res = optimize.nnls(X[:, cols], X[:, i])
        H[:, i] = sol
    rel_res = np.linalg.norm(X - np.dot(X[:, cols], H), 'fro')
    rel_res /= np.linalg.norm(X, 'fro')
    return H, rel_res

def ComputeNMF(data, colnorms, r):

    data = np.copy(data)
    colinv = np.linalg.pinv(np.diag(colnorms))

    A = np.dot(data, colinv)
    _, S, Vt = np.linalg.svd(A)
    A = np.dot(np.diag(S), Vt)
    cols = SPA(A, r)

    H, rel_res = NNLSFrob(data, cols)
    return cols, H, rel_res

def ParseMatrix(matpath):
	matrix = []
	with open(matpath, 'r') as f:
            for row in f:
                matrix.append([float(v) for v in row.split()[1:]])
	return np.array(matrix)

def ParseColnorms(colpath):
	norms = []
	with open(colpath, 'r') as f:
            for line in f:
                norms.append(float(line.split()[-1]))
	return norms

data = ParseMatrix(train_documents)
colnorms = ParseColnorms(train_categories)
r = 4
cols, H, rel_res = ComputeNMF(data, colnorms, r)
cols.sort()

print("Final result: ", rel_res)

# r is separation rank, X is dataset
