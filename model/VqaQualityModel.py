import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model


class GatedTanh():
    def __init__(self, input_shape, output_shape):
        self._tanh = layers.Dense(output_shape, input_shape=(input_shape,), activation='tanh')
        self._sigmoid = layers.Dense(output_shape, input_shape=(input_shape,), activation='sigmoid')
        self._multiply = layers.Multiply()

    def __call__(self, inputs):
        t = self._tanh(inputs)
        s = self._sigmoid(inputs)
        return self._multiply([t,s])

    
class VqaQualityModel():       
    def build_options(self):
        options = {}        
        options['vocab_size'] = 3797 
        options['embed_size'] = 300
        options['question_len'] = 14
        options['q_encoded_size'] = 512
        options['img_feat_wh'] = 14 if self.fmap_source == 'resnet152' else 10  # grid: 14; region (Detectron): 10
        options['img_feat_dim'] = 2048
        options['img_feat_shape'] = [options['img_feat_wh'], options['img_feat_wh'], options['img_feat_dim']]
        options['topdown_gated_size'] = 512
        options['ans_fc_dim'] = 1024
        options['drop_prob'] = 0.5

        options['batch_size'] = 256
        options['num_epochs'] = 20

        return options
            
    def __init__(self, fmap_source='detectron'):
        self.fmap_source = fmap_source
        options = self.build_options()
        self.options = options
        self.outputs = {}
        self.my_losses = {}
        
        with tf.compat.v1.variable_scope('vqa_quality'):            
            self.q_embed = layers.Embedding(options['vocab_size'], options['embed_size'], input_length=options['question_len'])
            self.q_gru = layers.GRU(options['q_encoded_size'], dropout=options['drop_prob']) 

            self.topdown_tanh = layers.Conv2D(options['topdown_gated_size'], 1, activation='tanh')
            self.topdown_sig = layers.Conv2D(options['topdown_gated_size'], 1, activation='sigmoid')
            self.topdown_multiply = layers.Multiply()
            self.topdown_conv = layers.Conv2D(1, 1)

            self.q_gated_tanh = GatedTanh(options['q_encoded_size'], options['q_encoded_size'])
            self.img_gated_tanh = GatedTanh(options['img_feat_dim'], options['q_encoded_size'])
            self.VQ_joint = layers.Multiply()

            self.ans_gated_tanh = GatedTanh(options['q_encoded_size'], options['ans_fc_dim'])
            self.ans_dense = layers.Dense(2, input_shape=(options['ans_fc_dim'],), activation='sigmoid') 
                   
        
    def build_graph(self, img_feat_input, q_input):
        assert q_input.get_shape().as_list()[1] == self.options['question_len'], "Wrong question length!"
        assert img_feat_input.get_shape().as_list()[1:] == self.options['img_feat_shape'], "Wrong feature shape!"        
        
        embedded_q = self.q_embed(q_input)
        q_mask = self.q_embed.compute_mask(q_input)
        encoded_q = self.q_gru(embedded_q, mask=q_mask)
        _encoded_q = layers.Reshape((1,1, self.options['q_encoded_size']))(encoded_q)
        encoded_q_tile = tf.tile(_encoded_q, [1, self.options['img_feat_shape'][0], self.options['img_feat_shape'][1], 1])
        
        concat_VQ = layers.Concatenate(axis=-1)([img_feat_input, encoded_q_tile])
        topdown_tanh_out = self.topdown_tanh(concat_VQ)
        topdown_sig_out = self.topdown_sig(concat_VQ)
        topdown_gated_out = self.topdown_multiply([topdown_tanh_out, topdown_sig_out])
        topdown_out = self.topdown_conv(topdown_gated_out)
        prob_attention = tf.exp(topdown_out) / tf.reduce_sum(tf.exp(topdown_out), axis=(1,2), keepdims=True)
        attention_feat = tf.reduce_sum(prob_attention * img_feat_input, axis=(1,2))
        
        gated_encoded_q = self.q_gated_tanh(encoded_q)
        gated_atten_feat = self.img_gated_tanh(attention_feat)        
        vq_feat = self.VQ_joint([gated_encoded_q, gated_atten_feat]) 
        
        pred_ans_rec = self.ans_dense(self.ans_gated_tanh(vq_feat))         
        self.outputs['ans_and_rec'] = pred_ans_rec        
        
        return self.outputs.copy()
      

    def compute_losses(self, answerability_label, recognizability_label):
        loss = K.binary_crossentropy(tf.concat((answerability_label, recognizability_label), axis=-1), self.outputs['ans_and_rec'])
        self.my_losses['total'] = tf.reduce_mean(loss, axis=-1)
        
        return self.my_losses.copy()
    
    
