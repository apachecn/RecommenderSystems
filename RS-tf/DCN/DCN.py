import numpy as np
import pandas as pd
import keras.backend as K
from keras import layers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, Embedding, Reshape, Add
from keras.layers import Flatten, merge, Lambda
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import random

def feature_generate(data):
    data, label, cate_columns, cont_columns = preprocessing(data)
    embeddings_tensors = []
    continuous_tensors = []
    for ec in cate_columns:
        layer_name = ec + '_inp'
        # For categorical features, we em-bed the features in dense vectors of dimension 6×(category cardinality)**(1/4)
        embed_dim = data[ec].nunique() if int(6 * np.power(data[ec].nunique(), 1/4)) > data[ec].nunique() \
            else int(6 * np.power(data[ec].nunique(), 1/4))
        t_inp, t_build = embedding_input(layer_name, data[ec].nunique(), embed_dim)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    for cc in cont_columns:
        layer_name = cc + '_in'
        t_inp, t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    inp_layer =  [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]
    inp_embed =  [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]
    return data, label, inp_layer, inp_embed

def embedding_input(name, n_in, n_out):
    inp = Input(shape = (1, ), dtype = 'int64', name = name)
    return inp, Embedding(n_in, n_out, input_length = 1)(inp)

def continous_input(name):
    inp = Input(shape=(1, ), dtype = 'float32', name = name)
    return inp, Reshape((1, 1))(inp)



# The optimal hyperparameter settings were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN
# Embed "Soil_Type" column (embedding dim == 15), we have 8 cross layers of size 29   
def fit(inp_layer, inp_embed, X, y):
    #inp_layer, inp_embed = feature_generate(X, cate_columns, cont_columns)
    input = merge(inp_embed, mode = 'concat')
    # deep layer
    for i in range(6):
        if i == 0:
            deep = Dense(272, activation='relu')(Flatten()(input))
        else:
            deep = Dense(272, activation='relu')(deep)

    # cross layer
    cross = CrossLayer(output_dim = input.shape[2].value, num_layer = 8, name = "cross_layer")(input)

    #concat both layers
    output = merge([deep, cross], mode = 'concat')
    output = Dense(y.shape[1], activation = 'softmax')(output)
    model = Model(inp_layer, output) 
    print(model.summary())
    plot_model(model, to_file = 'model.png', show_shapes = True)
    model.compile(Adam(0.01), loss = 'categorical_crossentropy', metrics = ["accuracy"])
    model.fit([X[c] for c in X.columns], y, batch_size = 256, epochs = 10)
    return model


def evaluate(X, y, model):
    y_pred = model.predict([X[c] for c in X.columns])
    acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y, 1)) / y.shape[0]
    print("Accuracy: ", acc)


# https://keras.io/layers/writing-your-own-keras-layers/
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape = [1, self.input_dim], initializer = 'glorot_uniform', name = 'w_' + str(i), trainable = True))
            self.bias.append(self.add_weight(shape = [1, self.input_dim], initializer = 'zeros', name = 'b_' + str(i), trainable = True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims = True), self.bias[i], x]))(input)
            else:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), input), 1, keepdims = True), self.bias[i], input]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)



# modify the embedding columns here
def preprocessing(data):
    # inverse transform one-hot to continuous column
    df_onehot = data[[col for col in data.columns.tolist() if "Soil_Type" in col]]
    #for i in df_onehot.columns.tolist():
    #    if df_onehot[i].sum() == 0:
    #        del df_onehot[i]
    data["Soil"] = df_onehot.dot(np.array(range(df_onehot.columns.size))).astype(int)
    data.drop([col for col in data.columns.tolist() if "Soil_Type" in col], axis = 1, inplace = True)
    label = np.array(OneHotEncoder().fit_transform(data["Cover_Type"].reshape(-1, 1)).todense())
    del data["Cover_Type"]
    cate_columns = ["Soil"]
    cont_columns = [col for col in data.columns if col != "Soil"]
    # Feature normilization
    scaler = StandardScaler()
    data_cont = pd.DataFrame(scaler.fit_transform(data[cont_columns]), columns = cont_columns)
    data_cate = data[cate_columns]
    data = pd.concat([data_cate, data_cont], axis = 1)
    return data, label, cate_columns, cont_columns



if __name__ == "__main__":
    data = pd.read_csv("data/covtype.csv")
    X, y, inp_layer, inp_embed = feature_generate(data)
    
    #random split train and test by 9:1
    train_index = random.sample(range(X.shape[0]), int(X.shape[0] * 0.9))
    test_index = list(set(range(X.shape[0])) - set(train_index))

    model = fit(inp_layer, inp_embed, X.iloc[train_index], y[train_index, :])
    evaluate(X.iloc[test_index], y[test_index, :], model)