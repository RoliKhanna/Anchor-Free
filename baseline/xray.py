
from nltk.corpus import reuters
import sys
import numpy as np
from scipy import optimize

# Loading data here
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

def col2norm(X):
    return np.sum(np.abs(X) ** 2,axis=0)

def xray(X, r):
    cols = []
    R = np.copy(X)
    while len(cols) < r:
        i = np.argmax(col2norm(X))
        # Loop until we choose a column that has not been selected.
        while True:
            p = np.random.random((X.shape[0], 1))
            scores = col2norm(np.dot(R.T, X)) / col2norm(X)
            scores[cols] = -1   # IMPORTANT
            best_col = np.argmax(scores)
            if best_col in cols:
                # Re-try
                continue
            else:
                cols.append(best_col)
                H, rel_res = NNLSFrob(X, cols)
                R = X - np.dot(X[:, cols] , H)
                break
    return cols

def GP_cols(data, r):
    votes = {}
    for row in data:
        min_ind = np.argmin(row)
        max_ind = np.argmax(row)
        for ind in [min_ind, max_ind]:
            if ind not in votes:
                votes[ind] = 1
            else:
                votes[ind] += 1

    votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in votes][0:r]

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

    _, S, Vt = np.linalg.svd(data)
    A = np.dot(np.diag(S), Vt)
    cols = xray(data, r)

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
