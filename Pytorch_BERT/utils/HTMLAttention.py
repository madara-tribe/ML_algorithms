# HTMLを作成する関数を実装
import torch 
from torch import nn
import torchtext
from utils.bert import BertTokenizer
# 単語分割用のTokenizerを用意
tokenizer_bert = BertTokenizer(
    vocab_file="./vocab/bert-base-uncased-vocab.txt", do_lower_case=True)




def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights, TEXT):
    "HTMLデータを作成する"

    # indexの結果を抽出
    sentence = batch.Text[0][index]  # 文章
    label = batch.Label[index]  # ラベル
    pred = preds[index]  # 予測

    # ラベルと予測結果を文字に置き換え
    if label == 0:
        label_str = "Negative"
    else:
        label_str = "Positive"

    if pred == 0:
        pred_str = "Negative"
    else:
        pred_str = "Positive"

    # 表示用のHTMLを作成する
    html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

    # Self-Attentionの重みを可視化。Multi-Headが12個なので、12種類のアテンションが存在
    for i in range(12):

        # indexのAttentionを抽出と規格化
        # 0単語目[CLS]の、i番目のMulti-Head Attentionを取り出す
        # indexはミニバッチの何個目のデータかをしめす
        attens = normlized_weights[index, i, 0, :]
        attens /= attens.max()

        html += '[BERTのAttentionを可視化_' + str(i+1) + ']<br>'
        for word, attn in zip(sentence, attens):

            # 単語が[SEP]の場合は文章が終わりなのでbreak
            if tokenizer_bert.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
                break

            # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
            html += highlight(tokenizer_bert.convert_ids_to_tokens(
                [word.numpy().tolist()])[0], attn)
        html += "<br><br>"

    # 12種類のAttentionの平均を求める。最大値で規格化
    all_attens = attens*0  # all_attensという変数を作成する
    for i in range(12):
        attens += normlized_weights[index, i, 0, :]
    attens /= attens.max()

    html += '[BERTのAttentionを可視化_ALL]<br>'
    for word, attn in zip(sentence, attens):

        # 単語が[SEP]の場合は文章が終わりなのでbreak
        if tokenizer_bert.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
            break

        # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
        html += highlight(tokenizer_bert.convert_ids_to_tokens(
            [word.numpy().tolist()])[0], attn)
    html += "<br><br>"

    return html