#!/usr/bin/env python
# coding: utf-8

# In[9]:


from LoadTrainedBert import retrun_config, AttrDict, return_loaded_state_dict, load_trained_bert
from utils.tokenizer import BasicTokenizer, WordpieceTokenizer
import numpy as np
import torch
from torch import nn
# BasicTokenizer, WordpieceTokenizerは、引用文献[2]そのままです
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
# これらはsub-wordで単語分割を行うクラスになります。
# vocabファイルを読み込み、
import collections
import torch.nn.functional as F



class BertTokenizer(object):
    '''BERT用の文章の単語分割クラスを実装'''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        vocab_file：ボキャブラリーへのパス
        do_lower_case：前処理で単語を小文字化するかどうか
        '''

        # ボキャブラリーのロード
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        # 分割処理の関数をフォルダ「utils」からimoprt、sub-wordで単語分割を行う
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # (注釈)上記の単語は途中で分割させない。これで一つの単語とみなす

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        '''文章を単語に分割する関数'''
        split_tokens = []  # 分割後の単語たち
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """分割された単語リストをIDに変換する関数"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """IDを単語に変換する関数"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens



    
def load_vocab(vocab_file):
    """text形式のvocabファイルの内容を辞書に格納します"""
    vocab = collections.OrderedDict()  # (単語, id)の順番の辞書変数
    ids_to_tokens = collections.OrderedDict()  # (id, 単語)の順番の辞書変数
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()

            # 格納
            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


# In[14]:


def check_trained_bert_work(net):

    """Bankの文脈による意味変化を単語ベクトルとして求める """
    # 文章1：銀行口座にアクセスしました。
    text_1 = "[CLS] I accessed the bank account. [SEP]"

    # 文章2：彼は敷金を銀行口座に振り込みました。
    text_2 = "[CLS] He transferred the deposit money into the bank account. [SEP]"

    # 文章3：川岸でサッカーをします。
    text_3 = "[CLS] We play soccer at the bank of the river. [SEP]"

    # 単語分割Tokenizerを用意
    tokenizer = BertTokenizer(
        vocab_file="./vocab/bert-base-uncased-vocab.txt", do_lower_case=True)

    # 文章を単語分割
    tokenized_text_1 = tokenizer.tokenize(text_1)
    tokenized_text_2 = tokenizer.tokenize(text_2)
    tokenized_text_3 = tokenizer.tokenize(text_3)

    print('===check input text===')
    print(tokenized_text_1)


    # In[37]:


    # 単語をIDに変換する
    indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
    indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
    indexed_tokens_3 = tokenizer.convert_tokens_to_ids(tokenized_text_3)

    # 各文章のbankの位置
    bank_posi_1 = np.where(np.array(tokenized_text_1) == "bank")[0][0]  # 4
    bank_posi_2 = np.where(np.array(tokenized_text_2) == "bank")[0][0]  # 8
    bank_posi_3 = np.where(np.array(tokenized_text_3) == "bank")[0][0]  # 6

    # seqId（1文目か2文目かは今回は必要ない）

    # リストをPyTorchのテンソルに
    tokens_tensor_1 = torch.tensor([indexed_tokens_1])
    tokens_tensor_2 = torch.tensor([indexed_tokens_2])
    tokens_tensor_3 = torch.tensor([indexed_tokens_3])

    # bankの単語id
    bank_word_id = tokenizer.convert_tokens_to_ids(["bank"])[0]

    print('===check input text vector===')
    print(tokens_tensor_1)

    # 文章をBERTで処理
    with torch.no_grad():
        encoded_layers_1, _ = net(tokens_tensor_1, output_all_encoded_layers=True)
        encoded_layers_2, _ = net(tokens_tensor_2, output_all_encoded_layers=True)
        encoded_layers_3, _ = net(tokens_tensor_3, output_all_encoded_layers=True)


    # bankの初期の単語ベクトル表現
    # これはEmbeddingsモジュールから取り出し、単語bankのidに応じた単語ベクトルなので3文で共通している
    bank_vector_0 = net.embeddings.word_embeddings.weight[bank_word_id]

    # 文章1のBertLayerモジュール1段目から出力されるbankの特徴量ベクトル
    bank_vector_1_1 = encoded_layers_1[0][0, bank_posi_1]

    # 文章1のBertLayerモジュール最終12段目から出力されるのbankの特徴量ベクトル
    bank_vector_1_12 = encoded_layers_1[11][0, bank_posi_1]

    # 文章2、3も同様に
    bank_vector_2_1 = encoded_layers_2[0][0, bank_posi_2]
    bank_vector_2_12 = encoded_layers_2[11][0, bank_posi_2]
    bank_vector_3_1 = encoded_layers_3[0][0, bank_posi_3]
    bank_vector_3_12 = encoded_layers_3[11][0, bank_posi_3]

    print('===trained BERTのコサイン類似度を計算===')

    print("bankの初期ベクトル と 文章1の1段目のbankの類似度：",
          F.cosine_similarity(bank_vector_0, bank_vector_1_1, dim=0))
    print("bankの初期ベクトル と 文章1の12段目のbankの類似度：",
          F.cosine_similarity(bank_vector_0, bank_vector_1_12, dim=0))

    print("文章1の1層目のbank と 文章2の1段目のbankの類似度：",
          F.cosine_similarity(bank_vector_1_1, bank_vector_2_1, dim=0))
    print("文章1の1層目のbank と 文章3の1段目のbankの類似度：",
          F.cosine_similarity(bank_vector_1_1, bank_vector_3_1, dim=0))

    print("文章1の12層目のbank と 文章2の12段目のbankの類似度：",
          F.cosine_similarity(bank_vector_1_12, bank_vector_2_12, dim=0))
    print("文章1の12層目のbank と 文章3の12段目のbankの類似度：",
          F.cosine_similarity(bank_vector_1_12, bank_vector_3_12, dim=0))
          


if __name__ == '__main__':
    config = retrun_config()
    config = AttrDict(config)
    loaded_state_dict = return_loaded_state_dict()
    net = load_trained_bert(config, loaded_state_dict)

    vocab_file = "./vocab/bert-base-uncased-vocab.txt"
    vocab, ids_to_tokens = load_vocab(vocab_file)
    print('vocab', len(vocab), 'ids_to_tokens', len(ids_to_tokens))
    check_trained_bert_work(net)


