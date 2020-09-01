import torch
from torch import nn

"""BertPoolerモジュール"""
class BertPooler(nn.Module):
    '''入力文章の1単語目[cls]の特徴量を変換して保持するためのモジュール'''

    def __init__(self, config):
        super(BertPooler, self).__init__()

        # 全結合層、'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 1単語目の特徴量を取得
        first_token_tensor = hidden_states[:, 0]

        # 全結合層で特徴量変換
        pooled_output = self.dense(first_token_tensor)

        # 活性化関数Tanhを計算
        pooled_output = self.activation(pooled_output)

        return pooled_output


