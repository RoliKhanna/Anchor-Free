
import numpy as np
import pandas as pd
from nltk.corpus import reuters
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading data here
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

# Initialsing LDA model
lda = LDA(n_components=1)
X_train = lda.fit_transform(train_documents, train_categories)
X_test = lda.transform(test_documents)

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train)
y_pred = classifier.predict(X_test)

# Evaluation parameters
cm = confusion_matrix(test_documents, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(test_documents, y_pred)))
