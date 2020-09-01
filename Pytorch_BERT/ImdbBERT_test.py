import random
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
import re
import string
from utils.bert import BertTokenizer
from utils.ImdbTcvReader import create_dataloder, tcv_reader, return_iter_batch
from utils.bert import BertTokenizer, load_vocab
from utils.bert import get_config, BertModel, set_learned_params
from IPython.display import HTML
from utils.HTMLAttention import mk_html

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)



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


class BertForIMDb(nn.Module):
    '''BERTモデルにIMDbのポジ・ネガを判定する部分をつなげたモデル'''

    def __init__(self, net_bert):
        super(BertForIMDb, self).__init__()

        # BERTモジュール
        self.bert = net_bert  # BERTモデル

        # headにポジネガ予測を追加
        # 入力はBERTの出力特徴量の次元、出力はポジ・ネガの2つ
        self.cls = nn.Linear(in_features=768, out_features=2)

        # 重み初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # BERTの基本モデル部分の順伝搬
        # 順伝搬させる
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)

        # 入力文章の1単語目[CLS]の特徴量を使用して、ポジ・ネガを分類します
        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, 768)  # sizeを[batch_size, hidden_sizeに変換
        out = self.cls(vec_0)

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out
        
        
def difine_bert_model():
    # モデル設定のJOSNファイルをオブジェクト変数として読み込みます
    config = get_config(file_path="./weights/bert_config.json")

    # BERTモデルを作成します
    net_bert = BertModel(config)

    # BERTモデルに学習済みパラメータセットします
    net_bert = set_learned_params(
        net_bert, weights_path="./weights/pytorch_model.bin")
    # モデル構築
    net = BertForIMDb(net_bert)
    return net






def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1

            # 開始時刻を保存
            t_epoch_start = time.time()
            t_iter_start = time.time()

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BertForIMDbに入力
                    outputs = net(inputs, token_type_ids=None, attention_mask=None,
                                  output_all_encoded_layers=False, attention_show_flg=False)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), duration, acc))
                            t_iter_start = time.time()

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))
            t_epoch_start = time.time()

    return net


# In[2]:


print("IMDbデータを読み込み、DataLoaderを作成（BERTのTokenizerを使用)")
train_ds, val_ds, test_ds, TEXT, LABEL = tcv_reader(formats='tsv', 
                                train_tsv='IMDb_train.tsv', test_tsv='IMDb_test.tsv')
train_dl, val_dl, test_dl = create_dataloder(train_ds, val_ds, test_ds, TEXT)
# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

print("Test Working")
batch = return_iter_batch(val_dl)


# In[3]:


print("感情分析用のBERTモデルを構築")
net = difine_bert_model()


# In[4]:


print('set up for bert finetune')

# 訓練モードに設定
net.train()

# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. まず全部を、勾配計算Falseにしてしまう
for name, param in net.named_parameters():
    param.requires_grad = False

# 2. 最後のBertLayerモジュールを勾配計算ありに変更
for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

# 3. 識別器を勾配計算ありに変更
for name, param in net.cls.named_parameters():
    param.requires_grad = True
    
# 最適化手法の設定

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 5e-5}
], betas=(0.9, 0.999))

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算

    
print('complete finshing network set up')


# In[5]:


print("TRAINING.....")
# モデルを学習させる関数を作成
num_epochs = 2
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)

print("Save model....")
# 学習したネットワークパラメータを保存します
save_path = './weights/bert_fine_tuning_IMDb.pth'
torch.save(net_trained.state_dict(), save_path)


# In[6]:


print("EVALATION")
# テストデータでの正解率を求める
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_trained.eval()   # モデルを検証モードに
net_trained.to(device)  # GPUが使えるならGPUへ送る

# epochの正解数を記録する変数
epoch_corrects = 0

for batch in tqdm(test_dl):  # testデータのDataLoader
    # batchはTextとLableの辞書オブジェクト
    # GPUが使えるならGPUにデータを送る
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = batch.Text[0].to(device)  # 文章
    labels = batch.Label.to(device)  # ラベル

    # 順伝搬（forward）計算
    with torch.set_grad_enabled(False):

        # BertForIMDbに入力
        outputs = net_trained(inputs, token_type_ids=None, attention_mask=None,
                              output_all_encoded_layers=False, attention_show_flg=False)

        loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

# 正解率
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

print('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset), epoch_acc))


# In[7]:


print("PREDICTION")

batch_size = 64
test_dl = torchtext.data.Iterator(
    test_ds, batch_size=batch_size, train=False, sort=False)
# BertForIMDbで処理

# ミニバッチの用意
batch = next(iter(test_dl))

# GPUが使えるならGPUにデータを送る
inputs = batch.Text[0].to(device)  # 文章
labels = batch.Label.to(device)  # ラベル

outputs, attention_probs = net_trained(inputs, token_type_ids=None, attention_mask=None,
                                       output_all_encoded_layers=False, attention_show_flg=True)

_, preds = torch.max(outputs, 1)  # ラベルを予測


print("Plot BERT Attention HTML")
index = 3  # 出力させたいデータ
html_output = mk_html(index, batch, preds, attention_probs, TEXT)  # HTML作成
HTML(html_output)  # HTML形式で出力



index = 61  # 出力させたいデータ
html_output = mk_html(index, batch, preds, attention_probs, TEXT)  # HTML作成
HTML(html_output)  # HTML形式で出力
