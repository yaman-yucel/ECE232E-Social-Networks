import numpy as np
import os
import networkx as nx
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import normalize

    
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras.backend as K

from utils_1layer import GraphConvLayer, GNNNodeClassifier

def convert_labels_to_int(y_labels):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(y_labels)
    return integer_labels


#1
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

                

all_data = shuffle(all_data,random_state=42)


#2
labels = []
nodes = []
X = []

for i,data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(elements[-1])
    X.append(elements[1:-1])
    nodes.append(elements[0])

X = np.array(X).astype('int32')
N = X.shape[0] 
F = X.shape[1]
Y = convert_labels_to_int(labels)

print('X shape: ', X.shape)

#parse the edge
edge_list=[]
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((e[0],e[1]))
edge_list = np.array(edge_list).astype('int32')
nodes = np.array(nodes).astype('int32')

#I need to zero value the node stuff
id_to_index = {}

for index, node_id in enumerate(nodes):
    id_to_index[node_id] = index
    
edge_list = [[id_to_index[node_id] for node_id in row] for row in edge_list]






# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
edges = tf.cast(edge_list, dtype = tf.dtypes.int32)
# Create a node features array of shape [num_nodes, num_features].
node_features = tf.cast(
    np.array(X), dtype=tf.dtypes.int32
)
node_features = np.array(node_features)
edge_weights = tf.ones(shape=edges.shape[1])

print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)
print('\nCategories: ', set(labels))

num_classes = len(set(labels))
print('\nNumber of classes: ', num_classes)

#3
graph_info = ((node_features,edges, edge_weights))


print(edges.shape)
print(node_features.shape)

hidden_units = [64]
learning_rate = 0.0079
dropout_rate = 0.81
num_epochs =  1000
batch_size = 125

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)



print("GNN output shape:", gnn_model([10,15,20]))

gnn_model.summary()



#5
def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=500, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history

X_train = []
y_train = []
counter = [0,0,0,0,0,0,0]

for index in range(0, 2708):
    c = Y[index]
    counter[c]+=1
    if counter[c] < 20:
        X_train.append(index)
        y_train.append(c)
X_train = np.array(X_train)        
y_train= np.array(y_train)

history = run_experiment(gnn_model, X_train , y_train)

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()
    
display_learning_curves(history)

X_test = np.array(range(1300,2000))
y_test = np.array(Y[1300:2000])
_, test_accuracy = gnn_model.evaluate(x=X_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")



