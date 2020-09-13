import tensorflow as tf
import numpy as np

class GroupLinearLayer(tf.keras.layers.Layer):
    
    def __init__(self, units, nRIM):
        super(GroupLinearLayer, self).__init__()
        self.units = units
        self.nRIM = nRIM
        
    def build(self, input_shape):
        # input_shape = (batch, [time,] nRIM, din)
        self.w = self.add_weight(name = 'group_linear_layer',
                                shape = (self.nRIM, int(input_shape[-1]), self.units),
                                initializer = 'random_normal',
                                trainable = True)
    def call(self, inputs):
        params = self.w
        out = tf.transpose(tf.matmul(tf.transpose(inputs, [1,0,2]), params),[1,0,2])
        return out


class GroupLSTMCell(tf.keras.layers.Layer):
    
    def __init__(self, units, nRIM):
        super(GroupLSTMCell, self).__init__()
        self.units = units
        self.nRIM = nRIM
        
    @property
    def state_size(self):
        return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))
    
    def build(self, input_shape):
        self.i2h_param = self.add_weight(name = 'group_lstm_i2h',
                                        shape = (self.nRIM, int(input_shape[-1]), self.units*4),
                                        initializer = 'uniform',
                                        trainable = True)
        self.h2h_param = self.add_weight(name = 'group_lstm_h2h',
                                        shape = (self.nRIM, self.units, self.units*4),
                                        initializer = 'uniform',
                                        trainable = True)

    def call(self, inputs, states):
        # inputs in shape [batch, nRIM, din]
        h,c = states
        preact_i = tf.transpose(tf.matmul(tf.transpose(inputs, [1,0,2]), self.i2h_param), [1,0,2])
        preact_h = tf.transpose(tf.matmul(tf.transpose(h,      [1,0,2]), self.h2h_param), [1,0,2])
        preact = preact_i + preact_h
        
        new_cell = tf.tanh(preact[:,:,:self.units])
        gates = tf.sigmoid(preact[:,:,self.units:])
        input_gate = gates[:,:,:self.units]
        forget_gate= gates[:,:,self.units:(self.units*2)]
        output_gate= gates[:,:,(self.units*2):]
        
        c_t = tf.multiply(c, forget_gate) + tf.multiply(input_gate, new_cell)
        h_t = tf.multiply(output_gate, tf.tanh(c_t))
        return h_t, (h_t, c_t)


class RIMCell(tf.keras.layers.Layer):
    
    def __init__(self, units, nRIM, k,
                num_input_heads, input_key_size, input_value_size, input_query_size, input_keep_prob,
                num_comm_heads, comm_key_size, comm_value_size, comm_query_size, comm_keep_prob):
        super(RIMCell, self).__init__()
        self.units = units
        self.nRIM = nRIM
        self.k = k
        
        self.num_input_heads = num_input_heads
        self.input_key_size = input_key_size
        self.input_value_size = input_value_size
        self.input_query_size = input_query_size
        self.input_keep_prob = input_keep_prob
        
        self.num_comm_heads = num_comm_heads
        self.comm_key_size = comm_key_size
        self.comm_value_size = comm_value_size
        self.comm_query_size = comm_query_size
        self.comm_keep_prob = comm_keep_prob
        
        assert input_key_size == input_query_size, 'input_key_size == input_query_size required'
        assert comm_key_size == comm_query_size, 'comm_key_size == comm_query_size required'
        
    @property
    def state_size(self):
        return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))

    def build(self, input_shape):
        self.key   = tf.keras.layers.Dense(units=self.num_input_heads*self.input_key_size,   activation=None, use_bias=True)
        self.value = tf.keras.layers.Dense(units=self.num_input_heads*self.input_value_size, activation=None, use_bias=True)
        self.query = GroupLinearLayer(units=self.num_input_heads*self.input_query_size, nRIM=self.nRIM)
        self.input_attention_dropout = tf.keras.layers.Dropout(rate = 1-self.input_keep_prob)
        
        self.rnn_cell = GroupLSTMCell(units=self.units, nRIM=self.nRIM)
        
        self.key_   = GroupLinearLayer(units=self.num_comm_heads*self.comm_key_size,   nRIM=self.nRIM)
        self.value_ = GroupLinearLayer(units=self.num_comm_heads*self.comm_value_size, nRIM=self.nRIM)
        self.query_ = GroupLinearLayer(units=self.num_comm_heads*self.comm_query_size, nRIM=self.nRIM)
        self.comm_attention_dropout = tf.keras.layers.Dropout(rate = 1-self.comm_keep_prob)
        self.comm_attention_output  = GroupLinearLayer(units=self.units, nRIM=self.nRIM)

        self.built = True
        
    def call(self, inputs, states, training = False):
        # inputs of shape (batch_size, input_feature_size)
        hs, cs = states
        rnn_inputs, mask = self.input_attention_mask(inputs, hs, training=training)
        
        h_old = hs*1.0
        c_old = cs*1.0
        
        _, (h_rnnout, c_rnnout) = self.rnn_cell(rnn_inputs, (hs, cs))
        
        h_new = tf.stop_gradient(h_rnnout*(1-mask)) + h_rnnout*mask
        
        h_comm = self.comm_attention(h_new, mask, training=training)
        
        h_update = h_comm*mask + h_old*(1-mask)
        c_update = c_rnnout*mask + c_old*(1-mask)
        return tf.reshape(h_update, [tf.shape(inputs)[0], -1]), (h_update, c_update)
    
    def input_attention_mask(self, x, hs, training=False):
        # x of shape (batch_size, input_feature_size)
        # hs of shape (batch_size, nRIM, hidden_size = units)
        
        xx = tf.stack([x, tf.zeros_like(x)], axis=1)
        
        key_layer   = self.key(xx)
        value_layer = self.value(xx)
        query_layer = self.query(hs)
        
        key_layer1   = tf.stack(tf.split(key_layer,   num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        value_layer1 = tf.stack(tf.split(value_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        query_layer1 = tf.stack(tf.split(query_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
        value_layer2 = tf.reduce_mean(value_layer1, axis=1)
        
        attention_scores1 = tf.matmul(query_layer1, key_layer1, transpose_b=True)/np.sqrt(self.input_key_size)
        attention_scores2 = tf.reduce_mean(attention_scores1, axis=1)
        
        signal_attention = attention_scores2[:,:,0]
        topk = tf.math.top_k(signal_attention, self.k)
        indices = topk.indices
        mesh = tf.meshgrid( tf.range(indices.shape[1]), tf.range(tf.shape(indices)[0]) )[1]
        full_indices = tf.reshape(tf.stack([mesh, indices], axis=-1), [-1,2])
        
        sparse_tensor = tf.sparse.SparseTensor(indices=tf.cast(full_indices, tf.int64),
                                              values=tf.ones(tf.shape(full_indices)[0]),
                                              dense_shape=[tf.shape(x)[0],self.nRIM])
        mask_ = tf.sparse.to_dense(sparse_tensor)
        mask  = tf.expand_dims(mask_, axis=-1)
        
        attention_prob = self.input_attention_dropout(tf.nn.softmax(attention_scores2, axis=-1), training=training)
        inputs = tf.matmul(attention_prob, value_layer2)
        inputs1 = inputs*mask
        return inputs1, mask
        
    def comm_attention(self, h_new, mask, training=False):
        # h_new of shape (batch_size, nRIM, hidden_size = units)
        # mask of shape (batch_size, nRIM, 1)
        
        comm_key_layer   = self.key_(h_new)
        comm_value_layer = self.value_(h_new)
        comm_query_layer = self.query_(h_new)
        
        comm_key_layer1   = tf.stack(tf.split(comm_key_layer,   num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
        comm_value_layer1 = tf.stack(tf.split(comm_value_layer, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
        comm_query_layer1 = tf.stack(tf.split(comm_query_layer, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
        
        comm_attention_scores = tf.matmul(comm_query_layer1, comm_key_layer1, transpose_b=True)/np.sqrt(self.comm_key_size)
        comm_attention_probs  = tf.nn.softmax(comm_attention_scores, axis=-1)
        
        comm_mask_ = tf.tile(tf.expand_dims(mask, axis=1), [1,self.num_comm_heads, 1, 1])
        
        comm_attention_probs1 = self.comm_attention_dropout(comm_attention_probs*comm_mask_, training=training)
        context_layer = tf.matmul(comm_attention_probs1, comm_value_layer1)
        context_layer1= tf.reshape(tf.transpose(context_layer, [0,2,1,3]), [tf.shape(h_new)[0], self.nRIM, self.num_comm_heads*self.comm_value_size])
        
        comm_out = self.comm_attention_output(context_layer1) + h_new
        return comm_out
        
        