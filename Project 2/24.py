#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:13:18 2023

@author: robertozturk
"""

import stellargraph as sg

# try:
#     sg.utils.validate_notebook_version("1.0.0rc1")
# except AttributeError:
#     raise ValueError(
#         f"This notebook requires StellarGraph version 1.0.0rc1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
#     ) from None
    
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn import svm

import os
import networkx as nx
import numpy as np
import pandas as pd

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph #feel free to use any other library of your choice
from stellargraph import datasets
from IPython.display import display, HTML

from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score

walk_length = 100

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load(largest_connected_component_only=True)
print(G.info())

nodes = G._nodes
rw = BiasedRandomWalk(G)

walks = rw.run(
    nodes=G.nodes(),  # root nodes
    length=walk_length,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    weighted=False,  # for weighted random walks
    seed=42,  # random seed fixed for reproducibility
)
print("Number of random walks: {}".format(len(walks)))

model = Word2Vec(
    walks, vector_size=128, window=5, min_count=0, sg=1, workers=1
)

node_ids = model.wv.index_to_key  # list of node IDs
node_embeddings = (
    model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets = node_subjects[[int(node_id) for node_id in node_ids]]
features = G.node_features(nodes = node_ids)




tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(node_embeddings)

alpha = 0.7
label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
node_colours = [label_map[target] for target in node_targets]

plt.figure(figsize=(10, 8))
plt.scatter(
    node_embeddings_2d[:, 0],
    node_embeddings_2d[:, 1],
    c=node_colours,
    cmap="jet",
    alpha=alpha,
)

X = node_embeddings
labels = np.array(node_targets)
y = np.array([label_map[label] for label in labels])
X2 = np.hstack((X,features))

X0_train = []
y_train = []
X1_train = []
X2_train = []
counter = [0,0,0,0,0,0,0]
train_indeces = []

for index in range(0, 2485):
    c = y[index]
    counter[c]+=1
    if counter[c] < 20:
        X0_train.append(X[index])
        X1_train.append(features[index])
        X2_train.append(X2[index])
        y_train.append(y[index])
        train_indeces.append(index)
        
X0_train = np.array(X0_train)        
y_train= np.array(y_train)
X1_train = np.array(X1_train)        
X2_train = np.array(X2_train)        



X0_test = np.delete(X, train_indeces, axis = 0)
X1_test = np.delete(features, train_indeces, axis = 0)
X2_test = np.delete(X2, train_indeces, axis = 0)
y_test = np.delete(y, train_indeces, axis = 0)


#clf = classifier, use kfold cross validation 
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
clf = svm.LinearSVC()
scores = cross_val_score(clf, X0_train, y_train, cv=5)
clf.fit(X0_train, y_train)

#prediction
y_pred = clf.predict(X0_test)
score0 = accuracy_score(y_test, y_pred)

print("Accuracy score for Word2Vec "+str(score0))

#clf = classifier, use kfold cross validation 
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
clf = svm.LinearSVC()
scores = cross_val_score(clf, X1_train, y_train, cv=5)
clf.fit(X1_train, y_train)

#prediction
y_pred = clf.predict(X1_test)
score1 = accuracy_score(y_test, y_pred)

print("Accuracy score for features "+str(score1))


#clf = classifier, use kfold cross validation 
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
clf = svm.LinearSVC()
scores = cross_val_score(clf, X2_train, y_train, cv=5)
clf.fit(X2_train, y_train)

#prediction
y_pred = clf.predict(X2_test)
score2 = accuracy_score(y_test, y_pred)

print("Accuracy score for W2V and features "+str(score2))






















