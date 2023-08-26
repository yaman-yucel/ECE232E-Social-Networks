# -*- coding: utf-8 -*-
import numpy as np
import os
import networkx as nx
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from numpy import dot
from numpy.linalg import norm
from numpy.random import choice, random

from collections import Counter
import matplotlib.pyplot as plt

all_data = []
all_edges = []

for root,dirs,files in os.walk('./cora'):
    for file in files:
        if '.content' in file:
            with open(os.path.join(root,file),'r') as f:
                all_data.extend(f.read().splitlines())
        elif 'cites' in file:
            with open(os.path.join(root,file),'r') as f:
                all_edges.extend(f.read().splitlines())

categories =  ['Reinforcement_Learning', 'Theory', 'Case_Based', 'Genetic_Algorithms', 'Probabilistic_Methods', 'Neural_Networks', 'Rule_Learning']
sorted(categories)
label_encoder = {}
i = 0
for cat in sorted(categories):
  label_encoder[cat] = i
  i +=1
label_encoder

#parse the data
labels = []
nodes = []
X = []
element_to_ind  = {}

for i,data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(label_encoder[elements[-1]])
    X.append(elements[1:-1])
    nodes.append(elements[0])
    element_to_ind[elements[0]]= i
X = np.array(X,dtype=int)
N = X.shape[0] #the number of nodes
F = X.shape[1] #the size of node features
print('X shape: ', X.shape)


#parse the edge
edge_list=[]
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((e[0],e[1]))

print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)
print('\nCategories: ', set(labels))

num_classes = len(set(labels))
print('\nNumber of classes: ', num_classes)

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)
G = nx.relabel_nodes(G, element_to_ind)
print('Graph info: ', nx.info(G))

nodes = list(G.nodes)
print(len(nodes))
print(list(G.neighbors(0)))

df = pd.DataFrame(list(zip(nodes, labels,X)),columns =['node', 'label','features'])
print(len(df))
df.head()

Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])
gcc_nodes = list(G.nodes)

df = df.loc[df['node'].isin(gcc_nodes)]
df['node'] = list(range(len(df))) #rename nodes 
df.head()

train = df.groupby('label', group_keys=False).apply(lambda x: x.sample(20))
G = nx.relabel_nodes(G, df['node'])

def create_transition_matrix(g):
    vs = list(g.nodes)
    n = len(vs)
    adj = nx.adjacency_matrix(g)
    transition_matrix = adj/adj.sum(axis=1)

    return transition_matrix

def random_walk(g, num_steps, start_node, transition_matrix = None):
    if transition_matrix is None:
        transition_matrix = create_transition_matrix(g)
    
    current_node = start_node
    
    for _ in range(num_steps):
        probabilities = np.array(transition_matrix[current_node]).flatten()
        
        next_node = choice(range(len(probabilities)), p=probabilities)
        
        current_node = next_node
    
    return current_node

seeds_dict = {predicted:list(train[train['label'] == predicted]['node']) for predicted in range(7)}

def random_walk_with_teleportation(g, num_steps, start_node,tp,predicted, transition_matrix = None):
    if transition_matrix is None:
        transition_matrix = create_transition_matrix(g)
        
    current_node = start_node
    
    for _ in range(num_steps):
        teleportation_chance = random()
        if teleportation_chance < tp:
            same_class_nodes = seeds_dict[predicted]
            next_node = choice(same_class_nodes)
        else:
            probabilities = np.array(transition_matrix[current_node]).flatten()
            next_node = choice(range(len(probabilities)), p=probabilities)
    
        current_node = next_node
    
    return current_node

#pagerank. NO teleportation, NO tfidf. 
transition_matrix = create_transition_matrix(G)

num_samples = 1000
num_walk_steps = 10

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk(G, num_walk_steps, start_point, transition_matrix)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1
      
count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)

#pagerank. WITH teleportation, NO tfidf. tp = .1
transition_matrix = create_transition_matrix(G)

num_samples = 1000
num_walk_steps = 10
tp = 0.1

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  #print (train_node,predicted)
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk_with_teleportation(G, num_walk_steps, start_point, tp, predicted, transition_matrix)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1

count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)

#pagerank. WITH teleportation, NO tfidf. tp = .2
transition_matrix = create_transition_matrix(G)

num_samples = 1000  
num_walk_steps = 10
tp = 0.2

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  #print (train_node,predicted)
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk_with_teleportation(G, num_walk_steps, start_point, tp, predicted, transition_matrix)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1

count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)

#tp = 0
vs = list(G.nodes)
n = len(vs)
adj = nx.adjacency_matrix(G)
transition = np.zeros((len(G.nodes), len(G.nodes)))

tf = adj / adj.sum(axis=1)
idf = np.log(n / (adj.sum(axis=0) + 1))

tfidf = np.multiply(tf, idf)

softmax_denominator = np.exp(tfidf).sum(axis=1)

for n1 in vs:
    for n2 in vs:
        if G.has_edge(n1, n2):
            cos_sim = np.dot(np.array(tfidf[n1]).flatten(), np.array(tfidf[n2]).flatten())
            cos_sim /= (np.linalg.norm(tfidf[n1]) * np.linalg.norm(tfidf[n2]))
            transition[n1, n2] = np.exp(cos_sim)
            
softmax_denominator = transition.sum(axis=1)
transition /= softmax_denominator[:, np.newaxis]


num_samples = 1000
num_walk_steps = 10
tp = 0

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  #print (train_node,predicted)
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk_with_teleportation(G, num_walk_steps, start_point, tp, predicted, transition)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1

count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)

#tp = .1
vs = list(G.nodes)
n = len(vs)
adj = nx.adjacency_matrix(G)
transition = np.zeros((len(G.nodes), len(G.nodes)))

tf = adj / adj.sum(axis=1)
idf = np.log(n / (adj.sum(axis=0) + 1))

tfidf = np.multiply(tf, idf)

softmax_denominator = np.exp(tfidf).sum(axis=1)

for n1 in vs:
    for n2 in vs:
        if G.has_edge(n1, n2):
            cos_sim = np.dot(np.array(tfidf[n1]).flatten(), np.array(tfidf[n2]).flatten())
            cos_sim /= (np.linalg.norm(tfidf[n1]) * np.linalg.norm(tfidf[n2]))
            transition[n1, n2] = np.exp(cos_sim)
            
softmax_denominator = transition.sum(axis=1)
transition /= softmax_denominator[:, np.newaxis]


num_samples = 1000  
num_walk_steps = 10
tp = 0.1

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  #print (train_node,predicted)
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk_with_teleportation(G, num_walk_steps, start_point, tp, predicted, transition)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1

count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)

#tp = .2
vs = list(G.nodes)
n = len(vs)
adj = nx.adjacency_matrix(G)
transition = np.zeros((len(G.nodes), len(G.nodes)))

tf = adj / adj.sum(axis=1)
idf = np.log(n / (adj.sum(axis=0) + 1))

tfidf = np.multiply(tf, idf)

softmax_denominator = np.exp(tfidf).sum(axis=1)

for n1 in vs:
    for n2 in vs:
        if G.has_edge(n1, n2):
            cos_sim = np.dot(np.array(tfidf[n1]).flatten(), np.array(tfidf[n2]).flatten())
            cos_sim /= (np.linalg.norm(tfidf[n1]) * np.linalg.norm(tfidf[n2]))
            transition[n1, n2] = np.exp(cos_sim)
            
softmax_denominator = transition.sum(axis=1)
transition /= softmax_denominator[:, np.newaxis]


num_samples = 1000  
num_walk_steps = 10
tp = .2

visiting_freq_label = []
for i in range(transition_matrix.shape[0]):
  visiting_freq_label.append([0,0,0,0,0,0,0])

visiting_freq = [0 for i in range(transition_matrix.shape[0])]


for train_node,predicted in zip(train['node'],train['label']):
  #print (train_node,predicted)
  for i in range(num_samples):
      start_point = train_node
      end_node = random_walk_with_teleportation(G, num_walk_steps, start_point, tp, predicted, transition)
      visiting_freq_label[end_node][predicted] += 1
      visiting_freq[end_node] +=1

count = 0 #these many nodes remain unvisited. 
for vf in visiting_freq:
  if vf ==0:
    count+=1
print('unvisited = ', count)
visiting_freq_label = np.asarray(visiting_freq_label)
preds = np.argmax(visiting_freq_label,axis = 1)
print(classification_report(df['label'], preds))
score = accuracy_score(df['label'], preds)
print(score)


