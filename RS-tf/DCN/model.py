import tensorflow as tf
import numpy as np
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class DCN(BaseEstimator, TransformerMixin):
    def __init__(self, cate_feature_size,field_size,numeric_feature_size,embedding_size=8,
                 deep_layers=[32,32],dropout_deep=[0.5,0.5,0.5],
                deep_layers_activation=tf.nn.relu,epoch=10,batch_size=256,
                 learning_rate=0.01,optimizer_type='adam',
                verbose=False,random_seed=2018,loss_type='logloss',
                eval_metric=roc_auc_score,l2_reg=0.0,cross_layer_num=3):
        assert loss_type in ["logloss", "rmse"], \
            "'logloss' for classification or 'rmse' for regression"

        self.cate_feature_size = cate_feature_size
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.train_result,self.valid_result = [],[]

        self._init_graph()
        
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_index = tf.placeholder(tf.int32, shape=[None,None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None,None], name='feat_value')
            self.numeric_value = tf.placeholder(tf.float32, shape=[None,None],name='num_value')
            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')
                
            self.weights = self._initialize_weights()
                
            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1,self.field_size,1])
            self.embeddings = tf.multiply(self.embeddings,feat_value)
                
            self.x0 = tf.concat([self.numeric_value,tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])], axis=1)
                
            # deep network
            self.y_deep = tf.nn.dropout(self.x0,self.dropout_keep_deep[0])
                
            for i in range(len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights['deep_layer_%d' % i]), self.weights['deep_bias_%d' % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])
                    
            # cross network
            self._x0 = tf.reshape(self.x0,(-1,self.total_size,1))
            x_l = self._x0
            for l in range(self.cross_layer_num):
                x_l = tf.tensordot(tf.matmul(self._x0,x_l, transpose_b=True),
                                       self.weights['cross_layer_%d' % l],1) + self.weights['cross_bias_%d' % l] + x_l
                self.cross_network_out = tf.reshape(x_l,(-1,self.total_size))
                    
                    
            # concat layer
            concat_input = tf.concat([self.cross_network_out, self.y_deep], axis=1)
                
            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])
                
            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label,self.out)
            elif self.loss_type == 'rmse':
                self.loss = tf.sqrt(tf.losses.mean_squared_error(self.label,self.out))
                    
                    
            # l2_reg
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['deep_layer_%d' % i])
                for i in range(len(self.cross_layer_num)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['cross_layer_%d' % i])
                        
                        
            # optimization
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(self.loss)
                    
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
                    
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                    
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
                    
            # init 
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
                
            # number of params
            total_parameters = 0
            for v in self.weights.values():
                shape = v.get_shape()
                value_params = 1
                for dim in shape:
                    value_params *= dim.value
                total_parameters += value_params
                    
            if self.verbose > 0:
                print('Parames: %d' % total_parameters)
                    
    def _initialize_weights(self):
        weights = dict()
        
        #embedding
        weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size,1],0.0,1.0),name='feature_bias')
        
        # deep network
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0 / (self.total_size + self.deep_layers[0]))
        
        weights['deep_layer_0'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(self.total_size,self.deep_layers[0])),dtype=np.float32)
        weights['deep_bias_0'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32) 
        
        
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.total_size + self.deep_layers[i]))
            # size = layers[i-1] * layers[i]
            weights['deep_layer_%d' % i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(self.deep_layers[i-1],self.deep_layers[i])),dtype=np.float32)
            #size = 1 * layers[i]
            weights['deep_bias_%d' % i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[i])),dtype=np.float32)
            
        # cross network
        
        for i in range(self.cross_layer_num):
            weights['cross_layer_%d' % i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(self.total_size,1)), dtype=np.float32)
            weights['cross_bias_%d' % i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(self.total_size,1)), dtype=np.float32)
            
        
        # Concat layers
        input_size = self.total_size + self.deep_layers[-1]
        
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        
        return weights
            
        
    def get_batch(self,Xi,Xv,Xv2,y,batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]
   

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self,a,b,c,d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        
        
    def predict(self,Xi,Xv,Xv2,y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.train_phase: True}
        
        loss = self.sess.run([self.loss], feed_dict=feed_dict)
        
        return loss
    
    def fit_on_batch(self,Xi,Xv,Xv2,y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.train_phase: True}
        
        loss, opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
        
        return loss
    
    def fit(self,cate_Xi_train,cate_Xv_train,numeric_Xv_train,y_train,
           cate_Xi_valid=None,cate_Xv_valid=None,numeric_Xv_valid=None,y_valid=None,
           early_stopping=False,refit=False):
        """
        :Xi_train: feature index of feature field of sample in the training set
        :Xv_train: feature value of feature field of sample in the training set; can be either binary or float
        :y_train: label of each sample in the training set
        :Xi_valid: feature indices of each sample in the validation set
        :Xv_valid: feature values of each sample in the validation set
        :y_valid: label of each sample in the validation set
        :early_stopping: early stopping or not
        :refit: refit the model on the train+valid dataset or not
        """
        
        print(len(cate_Xi_train))
        print(len(cate_Xv_train))
        print(len(numeric_Xv_train))
        print(len(y_train))
        
        has_valid = cate_Xv_valid is not None
        
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(cate_Xi_train,cate_Xv_train,numeric_Xv_train,y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch, numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train,cate_Xv_train,numeric_Xv_train,y_train,self.batch_size,1)
                
                self.fit_on_batch(cate_Xi_batch,cate_Xv_batch,numeric_Xv_batch,y_batch)
                
                
            if has_valid:
                y_valid = np.array(y_valid).reshape((-1,1))
                loss = self.predict(cate_Xi_valid,cate_Xv_valid,numeric_Xv_valid,y_valid)
                print('epoch: ',epoch, 'loss:',loss)