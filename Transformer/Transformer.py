from __future__ import print_function
import numpy as np 
import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import tensorflow as tf

from embedding import TokenEmbedding, AddPositionalEncoding
from Attention import SelfAttention, MultiheadAttention
from metric_layers import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization, padded_cross_entropy_loss, padded_accuracy

# !pip install sentencepiece
import sentencepiece as spm
import random
from typing import List, Sequence, Tuple

class BatchGenerator:
    def __init__(
            self,
            max_length=50,
            spm_model_path: str = 'spm_natsume.model'
    ) -> None:
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.bos = self.sp.piece_to_id('<s>')
        self.eos = self.sp.piece_to_id('</s>')
        self.pad = 0

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def load(self, file_path: str) -> None:
        with open(file_path) as f:
            lines = [line.strip() for line in f.readlines()]
        data = self._create_data(lines)
        return data
      
    def get_batch(self, data, batch_size = 128, shuffle=True):
        while True:
            if shuffle:
                random.shuffle(data)
            raw_batch_list = self._split(data, batch_size)
            for raw_batch in raw_batch_list:
                questions, answers = zip(*raw_batch)
                yield self._convert_to_array(questions), self._convert_to_array(answers)
                # yield encoder_input, decoder_input
                   
    def _create_data(self, lines: Sequence[str]) -> List[Tuple[List[int], List[int]]]:
        questions = [self._create_question(line) for line in lines[:-1]]
        answers = [self._create_answer(line) for line in lines[1:]]
        return list(zip(questions, answers))

    def _create_question(self, sentence) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        return ids[:self.max_length]

    def _create_answer(self, sentence: str) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        return [self.bos] + ids[:self.max_length - 2] + [self.eos]

    def _split(self, nd_list: Sequence, batch_size: int) -> List[List]:
        return [list(nd_list[i - batch_size:i]) for i in range(batch_size, len(nd_list) + 1, batch_size)]

    def _convert_to_array(self, id_list_list: Sequence[Sequence[int]]) -> np.ndarray:
        max_len = max([len(id_list) for id_list in id_list_list])

        return np.array([list(id_list) + [self.pad] * (max_len - len(id_list)) for id_list in id_list_list], dtype=np.int32,)

# Transformer(Encoder Decoder) setup

PAD_ID = 0
hopping_num = 4
head_num = 8
hidden_dim = 512
dropout_rate = 0.1
vocab_size = 50

encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')
decoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_input')


class Transformer_encoder:
    def __init__(self):
      self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
      self.add_position_embedding = AddPositionalEncoding()
      self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
      self.attention_block_list: List[List[tf.keras.models.Model]] = []

      for _ in range(hopping_num):
          attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
          ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
          self.attention_block_list.append([
              ResidualNormalizationWrapper(attention_layer, dropout_rate, name='self_attention_wrapper'),
              ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
          ])
      self.output_normalization = LayerNormalization()  

  
    def Encoder(self):

      enc_attention_mask = self.create_enc_attention_mask(encoder_input)

      embedded_input = self.token_embedding(encoder_input)
      embedded_input = self.add_position_embedding(embedded_input)
      query = self.input_dropout_layer(embedded_input, training=True) # query shape=(?, ?, 512)

      for i, layers in enumerate(self.attention_block_list):
        attention_layer, ffn_layer = tuple(layers)
        with tf.name_scope(f'hopping_{i}'):
            query = attention_layer(query, attention_mask=enc_attention_mask, training=True)
            query = ffn_layer(query, training=True)
      encoder_output = self.output_normalization(query)     
      return encoder_output, enc_attention_mask # output shape=(?, ?, 512)

    def create_enc_attention_mask(self, encoder_input):
      is_training = tf.placeholder(dtype=tf.bool, name='is_training')
      batch_size, length = tf.unstack(tf.shape(encoder_input))
      pad_array = tf.equal(encoder_input, PAD_ID)  # [batch_size, m_length]
      # shape broadcasting で [batch_size, head_num, (m|q)_length, m_length] になる
      return tf.reshape(pad_array, [batch_size, 1, 1, length]) 
    
    
class Transformer_decoder:
    def __init__(self):
      self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
      self.add_position_embedding = AddPositionalEncoding()
      self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
      self.attention_block_list: List[List[tf.keras.models.Model]] = []

      for _ in range(hopping_num):
          self_attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
          enc_dec_attention_layer = MultiheadAttention(hidden_dim, head_num, dropout_rate, name='enc_dec_attention')
          ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
          self.attention_block_list.append([
                      ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='self_attention_wrapper'),
                      ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='enc_dec_attention_wrapper'),
                      ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
                  ])
      self.output_normalization = LayerNormalization()
              # 注：本家ではここは TokenEmbedding の重みを転地したものを使っている
      self.output_dense_layer = tf.keras.layers.Dense(vocab_size, use_bias=False)
  
    def Decoder(self, encoder_output, enc_attention_mask):

      #input: shape = [batch_size, length]
      #param training: 学習時は True
      #return: shape = [batch_size, length, hidden_dim]
      # [batch_size, length, hidden_dim]
      decoder_inputs = decoder_input[:, :-1]

      dec_self_attention_mask = self.create_dec_self_attention_mask(decoder_inputs)

      embedded_input = self.token_embedding(decoder_inputs)
      embedded_input = self.add_position_embedding(embedded_input)
      query = self.input_dropout_layer(embedded_input, training=True) # query shape=(?, ?, 512)

      for i, layers in enumerate(self.attention_block_list):
        self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
        with tf.name_scope(f'hopping_{i}'):
            query = self_attention_layer(query, attention_mask=dec_self_attention_mask, training=True)
            query = enc_dec_attention_layer(query, memory=encoder_output,
                                                    attention_mask=enc_attention_mask, 
                                                    training=True)
            query = ffn_layer(query, training=True)

      query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
      decoder_output = self.output_dense_layer(query)
      return decoder_output  # [batch_size, length, vocab_size]=(?, ?, 50)

    def create_dec_self_attention_mask(self, decoder_input):
        batch_size, length = tf.unstack(tf.shape(decoder_input))
        pad_array = tf.equal(decoder_input, PAD_ID)  # [batch_size, m_length]
        pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

        autoregression_array = tf.logical_not(
          tf.matrix_band_part(tf.ones([length, length], dtype=tf.bool), -1, 0))  # 下三角が False
        autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])

        return tf.logical_or(pad_array, autoregression_array)

# save dir
log_dir = 'log'
ckpt_path = 'ck/model.ckpt'
os.makedirs(log_dir, exist_ok=True)

# load txt data
data_path='natsume.txt'
batch_generator = BatchGenerator()
data = batch_generator.load(data_path)

vocab_size = batch_generator.vocab_size # 8000

# load Transformer
t_encoder = Transformer_encoder()
encoder_output, enc_attention_mask = t_encoder.Encoder() # (?, ?, 512), (?, 1, 1, ?)
t_decoder = Transformer_decoder()
decoder_output = t_decoder.Decoder(encoder_output, enc_attention_mask) # shape=(?, ?, 50)

#loss, acc = loss_acc(decoder_output, decoder_input) 
logit = decoder_output
decoder_target = decoder_input[:, 1:]

prediction = tf.nn.softmax(logit, name='prediction')
xentropy, weights = padded_cross_entropy_loss(
          logit, decoder_target, smoothing=0.05, vocab_size=vocab_size)
loss = tf.identity(tf.reduce_sum(xentropy) / tf.reduce_sum(weights), name='loss')

accuracies, weights = padded_accuracy(logit, decoder_target)
acc = tf.identity(tf.reduce_sum(accuracies) / tf.reduce_sum(weights), name='acc') # <tf.Tensor 'acc:0' shape=() dtype=float32>

# optimizer
global_step = tf.train.get_or_create_global_step()

learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate,
    beta2=0.98,
)
optimize_op = optimizer.minimize(loss, global_step=global_step)




# train
saver = tf.train.Saver()

max_step = 100000
batch_size = 128
max_learning_rate = 0.0001
warmup_step = 4000

def get_learning_rate(step: int) -> float:
    rate = min(step ** -0.5, step * warmup_step ** -1.5) / warmup_step ** -0.5
    return max_learning_rate * rate

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0

for encoder_inputs, decoder_inputs in batch_generator.get_batch(data, batch_size=batch_size):
  feed = {encoder_input: encoder_inputs, 
          decoder_input : decoder_inputs,
          learning_rate: get_learning_rate(step + 1)}
  _, losses, accs, step = sess.run([optimize_op, loss, acc, global_step], feed_dict=feed)
  
  
  if step % 100 == 0:
      print(f'{step}: loss: {losses},\t acc: {accs}')
      saver.save(sess, ckpt_path, global_step=step)



