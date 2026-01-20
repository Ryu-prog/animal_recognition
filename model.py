# 以下を「model.py」に書き込み
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import streamlit as st
import io

# CLIPモデルとプロセッサをロード
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 動物園の動物リスト（日本語と英語）
animals_ja = [
    "ライオン", "ゾウ", "キリン", "トラ", "クマ", "パンダ", "ゴリラ", "チンパンジー", "オランウータン",
    "ヒョウ", "チーター", "カバ", "カンガルー", "コアラ", "ペンギン", "フラミンゴ", "オカメインコ",
    "ワニ", "カメレオン", "ヘビ", "サル", "イルカ", "クジラ", "アザラシ", "ペリカン", "トナカイ",
    "ヤギ", "羊", "牛", "馬", "鹿", "猿", "鳥", "魚", "その他の動物"
]
animals_en = [
    "lion", "elephant", "giraffe", "tiger", "bear", "panda", "gorilla", "chimpanzee", "orangutan",
    "leopard", "cheetah", "hippopotamus", "kangaroo", "koala", "penguin", "flamingo", "parrot",
    "crocodile", "chameleon", "snake", "monkey", "dolphin", "whale", "seal", "pelican", "reindeer",
    "goat", "sheep", "cow", "horse", "deer", "ape", "bird", "fish", "other animal"
]

@st.cache_data
def predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    # 画像をCLIP用に処理
    inputs = processor(text=animals_en, images=img, return_tensors="pt", padding=True)

    # モデルで予測
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 画像とテキストの類似度
    probs = logits_per_image.softmax(dim=1)  # 確率に変換

    # 上位3つの結果を取得
    top_probs, top_indices = torch.topk(probs[0], 3)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((animals_ja[idx], animals_en[idx], prob.item()))

    return results
    # 画像をCLIP用に処理
    inputs = processor(text=animals_en, images=img, return_tensors="pt", padding=True)

    # モデルで予測
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 画像とテキストの類似度
    probs = logits_per_image.softmax(dim=1)  # 確率に変換

    # 上位3つの結果を取得
    top_probs, top_indices = torch.topk(probs[0], 3)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((animals_ja[idx], animals_en[idx], prob.item()))

    return results
