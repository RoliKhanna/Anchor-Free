
import nltk
from nltk.corpus import reuters
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import cvxopt
import numpy

# Initialising data

train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
epsilon = 0.5   # trying random step size

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

# Initialising TFIDF Vector in problem matrix

size = int(len(train_documents)/100)
print("length: ", size)
vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
newArray = [train_documents[i] for i in range(size)]
vectorised_train_documents = vectorizer.fit_transform(newArray)

print("Completed computation of Co-occurrence matrix.")
co_matrix = vectorised_train_documents * vectorised_train_documents.T
co_matrix.setdiag(0)

print("Applying square root decomposition to Co-occurrence matrix... ")
Bo = csr_matrix(co_matrix.sqrt())
Bt = csr_matrix(Bo.transpose())

print("Beginning problem space")

# Initialise M here, definitely a different method to compute M here
M = numpy.identity(size)

# Looping until convergence
N = []
while True:

    for i in range(0, size):

        print("Executing iteration ", i)

        N = M
        a = []
        # Computing a here
        for j in range(0, size):
            step = (-1.)**(i+j)
            N = numpy.delete(M, j, axis=0)
            N = numpy.delete(N, i, axis=1)
            Nt = numpy.linalg.det(N)
            a.append(step*Nt)

        # a = numpy.asarray(a).astype(numpy.double).reshape(size,1)
        # Bo = cvxopt.matrix(Bo)
        # cvx_q = cvxopt.matrix(a)
        #
        # argmax = cvxopt.solvers.lp(cvx_q, Bo, [0,1])
        # argmin = cvxopt.solvers.lp(a, Bo, [0,-1])

        argmax = numpy.argmax(a)
        argmin = numpy.argmin(a)

        M[:,i] = numpy.argmax([argmax, argmin])   # check arguments here

    print("Completed iterations")

    if numpy.max(numpy.abs(numpy.linalg.det(N) - numpy.linalg.det(M))) <= epsilon:  # converged
        break

C = Bo.multiply(csr_matrix(M.transpose()))  #matrix multiplication logic here, final C
Ct = C.transpose()

# Impementing final determination of C, E

el1 = (Ct * C).todense()
inv = numpy.linalg.pinv(el1)
inv1 = inv * Ct
inv11 = inv1 * co_matrix * C
final = inv11 * inv     # final E

print("Final matrix C: ", C)
print("Final matrix E: ", final)
