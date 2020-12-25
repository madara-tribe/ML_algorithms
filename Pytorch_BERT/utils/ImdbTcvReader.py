"""IMDbデータを読み込み、DataLoaderを作成（BERTのTokenizerを使用）"""
# 前処理と単語分割をまとめた関数を作成
import re
import string
from utils.bert import BertTokenizer, load_vocab
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
import random

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 単語分割用のTokenizerを用意
tokenizer_bert = BertTokenizer(
    vocab_file="./vocab/bert-base-uncased-vocab.txt", do_lower_case=True)


def tcv_reader(formats='tsv', train_tsv='IMDb_train.tsv', test_tsv='IMDb_test.tsv'):
    def preprocessing_text(text):
        '''IMDbの前処理'''
        # 改行コードを消去
        text = re.sub('<br />', '', text)

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        # ピリオドなどの前後にはスペースを入れておく
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text
    
    # 前処理と単語分割をまとめた関数を定義
    # 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
    def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
        text = preprocessing_text(text)
        ret = tokenizer(text)  # tokenizer_bert
        return ret
    
    
    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    max_length = 256
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length, 
                            init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # (注釈)：各引数を再確認
    # sequential: データの長さが可変か？文章は長さがいろいろなのでTrue.ラベルはFalse
    # tokenize: 文章を読み込んだときに、前処理や単語分割をするための関数を定義
    # use_vocab：単語をボキャブラリーに追加するかどうか
    # lower：アルファベットがあったときに小文字に変換するかどうか
    # include_length: 文章の単語数のデータを保持するか
    # batch_first：ミニバッチの次元を先頭に用意するかどうか
    # fix_length：全部の文章を指定した長さと同じになるように、paddingします
    # init_token, eos_token, pad_token, unk_token：文頭、文末、padding、未知語に対して、どんな単語を与えるかを指定
    # フォルダ「data」から各tsvファイルを読み込みます
 
    # preprocessing as BERT data, taking few minutes 
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='./data/', train=train_tsv,
        test=test_tsv, format=formats,
        fields=[('Text', TEXT), ('Label', LABEL)])

    # torchtext.data.Datasetのsplit関数で訓練データとvalidationデータを分ける
    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8, random_state=random.seed(1234))
               
    return train_ds, val_ds, test_ds, TEXT, LABEL


def create_dataloder(train_ds, val_ds, test_ds, TEXT, vocab_path="./vocab/bert-base-uncased-vocab.txt"):

    batch_size = 32  # BERTでは16、32あたりを使用する
    # BERTはBERTが持つ全単語でBertEmbeddingモジュールを作成しているので、ボキャブラリーとしては全単語を使用します
    # そのため訓練データからボキャブラリーは作成しません

    # まずBERT用の単語辞書を辞書型変数に用意します


    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=vocab_path)


    # このまま、TEXT.vocab.stoi= vocab_bert (stoiはstring_to_IDで、単語からIDへの辞書)としたいですが、
    # 一度bulild_vocabを実行しないとTEXTオブジェクトがvocabのメンバ変数をもってくれないです。
    # （'Field' object has no attribute 'vocab' というエラーをはきます）

    # 1度適当にbuild_vocabでボキャブラリーを作成してから、BERTのボキャブラリーを上書きします
    TEXT.build_vocab(train_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert
    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）

    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False)
    
    return train_dl, val_dl, test_dl


def return_iter_batch(val_dl):
    # 動作確認 検証データのデータセットで確認
    batch = next(iter(val_dl))
    print('batch.TEXT', batch.Text)
    print('batch.LABEL', batch.Label)
    
    text_minibatch_1 = (batch.Text[0][1]).numpy()
    # IDを単語に戻す
    text = tokenizer_bert.convert_ids_to_tokens(text_minibatch_1)
    print('check mini batch first sentences')
    print(text)
    return batch

"""
# 単語分割用のTokenizerを用意
tokenizer_bert = BertTokenizer(
    vocab_file="./vocab/bert-base-uncased-vocab.txt", do_lower_case=True)

train_ds, val_ds, test_ds, TEXT, LABEL = tcv_reader(formats='tsv', 
                             train_tsv='IMDb_train.tsv', test_tsv='IMDb_test.tsv')

train_dl, val_dl, test_dl = create_dataloder(train_ds, val_ds, test_ds)
if __name__ == '__main__':
    return_iter_batch(val_dl)
"""