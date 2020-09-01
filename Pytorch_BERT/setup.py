import os
import urllib.request
import zipfile
import tarfile
import glob
import io
import string

def create_dir():
    print("creating folder....")
    data_dir = "data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    vocab_dir = "./vocab/"
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)



def download_vocab():
    print("downloading vovab....")
    # 単語集：ボキャブラリーをダウンロード
    # 'bert-base-uncased': 
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

    save_path="./vocab/bert-base-uncased-vocab.txt"
    url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    urllib.request.urlretrieve(url, save_path)


def download_trained_bert():
    print("downloading trained bert....")
    # BERTの学習済みモデル 'bert-base-uncased'
    # https://github.com/huggingface/pytorch-pretrained-BERT/
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz

    # ダウンロード
    save_path = "./weights/bert-base-uncased.tar.gz"
    url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
    urllib.request.urlretrieve(url, save_path)

    # 解凍
    archive_file = "./weights/bert-base-uncased.tar.gz"  # Uncasedは小文字化モードという意味です
    tar = tarfile.open(archive_file, 'r:gz')
    tar.extractall('./weights/')  # 解凍
    tar.close()  # ファイルをクローズ
    # フォルダ「weights」に「pytorch_model.bin」と「bert_config.json」ができる


def doenload_imdb():
    print("downloading imdb dataset....")
    # IMDbデータセットをダウンロード。30秒ほどでダウンロードできます
    target_dir_path="./data/"

    if not os.path.exists(target_dir_path):
        os.mkdir(target_dir_path)

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./data/aclImdb_v1.tar.gz"
    urllib.request.urlretrieve(url, save_path)

    # './data/aclImdb_v1.tar.gz'の解凍　1分ほどかかります

    # tarファイルを読み込み
    tar = tarfile.open('./data/aclImdb_v1.tar.gz')
    tar.extractall('./data/')  # 解凍
    tar.close()  # ファイルをクローズ
    

# tsvファイルを作成
def create_imdb_tsv():
    def imdb_train():
        f = open('./data/IMDb_train.tsv', 'w')

        path = './data/aclImdb/train/pos/'
        for fname in glob.glob(os.path.join(path, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as ff:
                text = ff.readline()

                # タブがあれば消しておきます
                text = text.replace('\t', " ")

                text = text+'\t'+'1'+'\t'+'\n'
                f.write(text)

        path = './data/aclImdb/train/neg/'
        for fname in glob.glob(os.path.join(path, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as ff:
                text = ff.readline()

                # タブがあれば消しておきます
                text = text.replace('\t', " ")

                text = text+'\t'+'0'+'\t'+'\n'
                f.write(text)

        f.close()
    
    def imdb_test():
        # テストデータの作成

        f = open('./data/IMDb_test.tsv', 'w')

        path = './data/aclImdb/test/pos/'
        for fname in glob.glob(os.path.join(path, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as ff:
                text = ff.readline()

                # タブがあれば消しておきます
                text = text.replace('\t', " ")

                text = text+'\t'+'1'+'\t'+'\n'
                f.write(text)


        path = './data/aclImdb/test/neg/'

        for fname in glob.glob(os.path.join(path, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as ff:
                text = ff.readline()

                # タブがあれば消しておきます
                text = text.replace('\t', " ")

                text = text+'\t'+'0'+'\t'+'\n'
                f.write(text)

        f.close()
    print('creating imdb train/test tsv file....')
    imdb_train()
    imdb_test()


if __name__ == '__main__':
    create_dir()
    download_vocab()
    download_trained_bert()
    doenload_imdb()
    create_imdb_tsv()