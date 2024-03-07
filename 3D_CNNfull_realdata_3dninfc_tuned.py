from __future__ import division, print_function, absolute_import

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, BatchNormalization, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.data import Dataset
import sys

import tensorflow as tf

print(tf.version.VERSION)

from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
sns.set_style('white')

from sklearn.metrics import confusion_matrix, accuracy_score

# Hyper Parameter
batch_size = 96
epochs = 200

hdf5_file = h5py.File("testhdf5train.h5", mode='r')
class_weights = np.array(hdf5_file['class_weights'][:])
yclass = len(class_weights)

# Set up TensorBoard
tensorboard = TensorBoard()
pad = 4
lenpad = 2*pad + 1

#https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
class SequenceExample(Sequence):
    
    def __init__(self, x_in, y_in, batch_size, shuffle=False):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        global strategy
        #print(strategy.cluster_resolver.task_type)
        #print(strategy.cluster_resolver.task_id)
        #print(index)
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        gc.collect()
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Conv2D layer
def Conv(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=None, strides=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same',
                      activation=activation, input_shape=input_shape, strides=strides)
    elif strides:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same',
                      activation=activation, strides=strides)
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)

# Define Model
def CNN(input_dim, num_classes):
    model = Sequential()
    
    model.add(Conv(32, (3,3,3), input_shape=input_dim, strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv(112, (1,1,1)))
    model.add(BatchNormalization())
    model.add(Conv(96, (1,1,1)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Dropout(0.2))

    model.add(Conv(80, (6,6,6), strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv(80, (1,1,1)))
    model.add(BatchNormalization())
    model.add(Conv(80, (1,1,1)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Dropout(0.2))

    model.add(Conv(496, (5,5,5), strides=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv(528, (1,1,1)))
    model.add(BatchNormalization())
    model.add(Conv(512, (1,1,1)))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Train Model
def train(optimizer, scheduler):
    global model

    print("Training...")
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) , loss = CategoricalFocalCrossentropy(), metrics=["accuracy"])

    class_weight = dict(list(enumerate(class_weights)))

    X_train=np.array(hdf5_file['X_train'][:])
    y_train=np.array(hdf5_file['y_train'][:])

    data=SequenceExample(X_train, y_train, batch_size)

    model.fit(x=data, epochs=epochs,
              verbose=2, callbacks=[scheduler, tensorboard],
              class_weight=class_weight, workers=8, use_multiprocessing=True)

def evaluate():
    global model

    X_test = np.array(hdf5_file['X_test'][:])
    y_test = np.array(hdf5_file['y_test'][:])

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    print(accuracy_score(pred,y_test))
    # Heat Map
    array = confusion_matrix(y_test, pred)
    cm = pd.DataFrame(array, index = range(yclass), columns = range(yclass))
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True,fmt='g')
    plt.show()
    plt.savefig("confmatrix_3dninfc_tuned.png")

    return pred;

def save_model():
    global model
    global strategy

    model_json = model.to_json()
    with open('model/model_3D' + str(strategy.cluster_resolver.task_id) + '.json', 'w') as f:
        f.write(model_json)

    model.save_weights('model/model_3D' + str(strategy.cluster_resolver.task_id) + '.h5')

    print('Model Saved.')

def load_model():
    f = open('model/model_3D.json', 'r')
    model_json = f.read()
    f.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model/model_3D.h5')

    print("Model Loaded.")
    return loaded_model

def set_tf_config(resolver, environment=None):
    """
    Set the TF_CONFIG env variable from the given cluster resolver.
    """
    cfg = {
    "cluster": resolver.cluster_spec().as_dict(),
    "task": {
        "type": resolver.get_task_info()[0],
        "index": resolver.get_task_info()[1],
    },
    "rpc_layer": resolver.rpc_layer,
    }
    if environment:
        cfg["environment"] = environment
    os.environ["TF_CONFIG"] = json.dumps(cfg)

if __name__ == '__main__':

    import tensorflow as tf
    import json
    import os
    
    #strategy = tf.distribute.MirroredStrategy()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Create the SlurmClusterResolver object
    slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    config = slurm_resolver._task_configuration
    # workaround a slurm resolver bug - see:
    # https://github.com/tensorflow/tensorflow/issues/49956
    # https://github.com/tensorflow/tensorflow/pull/58033
    fixconfig = {}
    for host in config:
        (a,b,c) = host.split("-")
        newhost = "{}-{}-{}".format(a,b,c.zfill(2))
        fixconfig[newhost] = config[host]
    slurm_resolver._task_configuration = fixconfig
    #print(slurm_resolver.cluster_spec())
    # Setup the SlurmClusterResolver
    set_tf_config(slurm_resolver)
    # Setup the GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.RING
    )
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options,
    cluster_resolver=slurm_resolver
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    y_train=np.array(hdf5_file['y_train'][:])
    resind=np.array(hdf5_file['resind'][:])
    with strategy.scope():

        optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
        scheduler = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

        model = CNN((lenpad,lenpad,lenpad,2),yclass)

        train(optimizer, scheduler)
        pred = evaluate()
        save_model()
        pred = pred.reshape(-1,1)
        voxels = np.concatenate((np.argmax(y_train,axis=1).reshape(-1,1),pred))
        np.savetxt('testarr2_3dninfc_tuned.txt',np.hstack((resind,voxels)))
        print('finished')
