import torch
from torch import nn

class BertEmbeddings(nn.Module):
    """文章の単語ID列と、1文目か2文目かの情報を、埋め込みベクトルに変換する
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # 3つのベクトル表現の埋め込み

        # Token Embedding：単語IDを単語ベクトルに変換、
        # vocab_size = 30522でBERTの学習済みモデルで使用したボキャブラリーの量
        # hidden_size = 768 で特徴量ベクトルの長さは768
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        # （注釈）padding_idx=0はidx=0の単語のベクトルは0にする。BERTのボキャブラリーのidx=0が[PAD]である。

        # Transformer Positional Embedding：位置情報テンソルをベクトルに変換
        # Transformerの場合はsin、cosからなる固定値だったが、BERTは学習させる
        # max_position_embeddings = 512　で文の長さは512単語
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding：文章の1文目、2文目の情報をベクトルに変換
        # type_vocab_size = 2
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # 作成したLayerNormalization層
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids： [batch_size, seq_len]の文章の単語IDの羅列
        token_type_ids：[batch_size, seq_len]の各単語が1文目なのか、2文目なのかを示すid
        '''

        # 1. Token Embeddings
        # 単語IDを単語ベクトルに変換
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        # token_type_idsがない場合は文章の全単語を1文目として、0にする
        # そこで、input_idsと同じサイズのゼロテンソルを作成
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding：
        # [0, 1, 2 ・・・]と文章の長さだけ、数字が1つずつ昇順に入った
        # [batch_size, seq_len]のテンソルposition_idsを作成
        # position_idsを入力して、position_embeddings層から768次元のテンソルを取り出す
        seq_length = input_ids.size(1)  # 文章の長さ
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayerNorm(nn.Module):
    """LayerNormalization層 """

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

