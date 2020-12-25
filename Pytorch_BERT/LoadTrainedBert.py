import math
import numpy as np
import json
from attrdict import AttrDict
import torch
from torch import nn

from layers.BERTModules import BertLayer, BertSelfOutput, gelu, BertIntermediate, BertOutput, BertEncoder
from layers.BertPooler import BertPooler
from layers.Attention import BertAttention, BertSelfAttention
from layers.Embeddings import BertEmbeddings, BertLayerNorm


""" BERT model connected all layers"""
class BertModel(nn.Module):
    '''モジュールを全部つなげたBERTモデル'''

    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3つのモジュールを作成
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Attentionのマスクと文の1文目、2文目のidが無ければ作成する
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # マスクの変形　[minibatch, 1, 1, seq_length]にする
        # 後ほどmulti-head Attentionで使用できる形にしたいので
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # マスクは0、1だがソフトマックスを計算したときにマスクになるように、0と-infにする
        # -infの代わりに-10000にしておく
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 順伝搬させる
        # BertEmbeddinsモジュール
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayerモジュール（Transformer）を繰り返すBertEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPoolerモジュール
        # encoderの一番最後のBertLayerから出力された特徴量を使う
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output



# In[3]:



def retrun_config(config_file = "./weights/bert_config.json"):
    # 設定をconfig.jsonから読み込み、JSONの辞書変数をオブジェクト変数に変換
    json_file = open(config_file, 'r')
    config = json.load(json_file)
    print('===check config output===')
    print(config)
    return config

def check_layers_work(config):
    print('===Check each layers work===')

    # 入力の単語ID列、batch_sizeは2つ
    input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
    print("入力の単語ID列のテンソルサイズ：", input_ids.shape)

    # マスク
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    print("入力のマスクのテンソルサイズ：", attention_mask.shape)

    # 文章のID。2つのミニバッチそれぞれについて、0が1文目、1が2文目を示す
    token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    print("入力の文章IDのテンソルサイズ：", token_type_ids.shape)


    # BERTの各モジュールを用意
    embeddings = BertEmbeddings(config)
    encoder = BertEncoder(config)
    pooler = BertPooler(config)

    # マスクの変形　[batch_size, 1, 1, seq_length]にする
    # Attentionをかけない部分はマイナス無限にしたいので、代わりに-10000をかけ算しています
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    print("拡張したマスクのテンソルサイズ：", extended_attention_mask.shape)

    # 順伝搬する
    out1 = embeddings(input_ids, token_type_ids)
    print("BertEmbeddingsの出力テンソルサイズ：", out1.shape)

    out2 = encoder(out1, extended_attention_mask)
    # out2は、[minibatch, seq_length, embedding_dim]が12個のリスト
    print("BertEncoderの最終層の出力テンソルサイズ：", out2[0].shape)

    out3 = pooler(out2[-1])  # out2は12層の特徴量のリストになっているので一番最後を使用
    print("BertPoolerの出力テンソルサイズ：", out3.shape)

def check_full_bert_work(config):
    print('===Check full connected BERT model working===')
    # inputs
    input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

    # BERTモデルを作る
    net = BertModel(config)

    # 順伝搬させる
    encoded_layers, pooled_output, attention_probs = net(
        input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

    print("encoded_layersのテンソルサイズ：", encoded_layers.shape)
    print("pooled_outputのテンソルサイズ：", pooled_output.shape)
    print("attention_probsのテンソルサイズ：", attention_probs.shape)
    
    
def return_loaded_state_dict(weights_path = "./weights/pytorch_model.bin"):
    print('===output trained bert parameter name===')
    # load trained weight
    loaded_state_dict = torch.load(weights_path)

    for s in loaded_state_dict.keys():
        print(s)
    return loaded_state_dict


# In[16]:


def load_trained_bert(config, loaded_state_dict): 
    net = BertModel(config)
    net.eval()

    # 現在のネットワークモデルのパラメータ名
    param_names = []  # パラメータの名前を格納していく
    print('=== trained BERT parameters name ===')
    for name, param in net.named_parameters():
        print(name)
        param_names.append(name)
        
    # state_dictの名前が違うので前から順番に代入する
    # 今回、パラメータの名前は違っていても、対応するものは同じ順番になっています

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = net.state_dict().copy()

    # 新たなstate_dictに学習済みの値を代入
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる
        print(str(key_name)+"→"+str(name))  # 何から何に入ったかを表示

        # 現在のネットワークのパラメータを全部ロードしたら終える
        if index+1 >= len(param_names):
            break

    # 新たなstate_dictを実装したBERTモデルに与える
    net.load_state_dict(new_state_dict)
    return net


if __name__ == '__main__':
    # 辞書変数をオブジェクト変数に
    config = retrun_config()
    config = AttrDict(config)
    print(config.hidden_size)
    check_layers_work(config)
    check_full_bert_work(config)