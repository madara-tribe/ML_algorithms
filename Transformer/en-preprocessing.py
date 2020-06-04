!pip install sentencepiece
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU, Embedding
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '<s>' + target_text + '</s>'
    input_texts.append(input_text)
    target_texts.append(target_text)
    
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

class SentenceVector:
  
  def load_data(self, texts_sentence):
    text_vector =self.id_vector(texts_sentence)
    return self.padding(text_vector)

  def sentence2words(self, sentence):
    stopwords = ["i", "a", "an", "the", "and", "or", "if", "is", "are", "am", "it", "this", "that", "of", "from", "in", "on"]
    sentence = sentence.lower() # 小文字化
    sentence = sentence.replace("\n", "") # 改行削除
    sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~]"), " ", sentence) # 記号をスペースに置き換え
    sentence = sentence.split(" ") # スペースで区切る
    sentence_words = []
    for word in sentence:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        sentence_words.append(word)        
    return sentence_words
  
  def id_vector(self, texts):
    words = {}
    for sentence in texts:
      #sentence_words = sentence2words(sentence)
      for word in sentence:
        if word not in words:
            words[word] = len(words)+1
    
    # 文章を単語ID配列にする
    data_x_vec = []
    for sentence in texts:
      #sentence_words = sentence2words(sentence)
      sentence_ids = []
      for word in sentence:
        sentence_ids.append(words[word])
      data_x_vec.append(sentence_ids)
    
    return data_x_vec
  
  def padding(self, data_x_vec):
    # 文章の長さを揃えるため、0でパディングする
    max_sentence_size = 0
    for sentence_vec in data_x_vec:
      if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)
    for sentence_ids in data_x_vec:
      while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, 0) # 先頭に追加
    return np.array(data_x_vec, dtype="int32")

encorders=SentenceVector()
encoder_input_data = encorders.load_data(input_texts)
encoder_input_data = encoder_input_data.reshape(len(encoder_input_data),
                                                max_encoder_seq_length)
decoders = SentenceVector()
target_texts = decoders.load_data(target_texts)
target_texts = target_texts.reshape(len(encoder_input_data),
                                                max_decoder_seq_length)
decoder_input_data = target_texts[:, :-1]
decoder_target_data = target_texts[:, 1:]

print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)
