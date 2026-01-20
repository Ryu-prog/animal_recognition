import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import io

st.sidebar.title("動物認識アプリ")
st.sidebar.write("生成AIを使って動物園の写真に写っている動物を判定します。")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))

# まず img_file を取得する（ここが最重要）
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
else:
    img_file = st.camera_input("カメラで撮影")

# img_file が None のときは何もしない
if img_file is None:
    st.stop()

with st.spinner("推定中..."):

    # 画像の読み込み
    if img_source == "カメラで撮影":
        img = Image.open(img_file)
    else:
        img = Image.open(io.BytesIO(img_file.getvalue()))

    img = img.convert("RGB")

    st.image(img, caption="対象の画像", width=480)

    # 予測
    results = predict(img)

    # 結果の表示
    st.subheader("判定結果")
    n_top = 3
    for result in results[:n_top]:
        st.write(f"{round(result[2]*100, 2)}% の確率で {result[0]} です。")

    # ダウンロードボタン
    top_result = results[0]
    if img_source == "画像をアップロード":
        ext = img_file.name.split('.')[-1]
    else:
        ext = 'jpg'
    new_name = f"{top_result[0]}.{ext}"
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    st.download_button(label="画像をダウンロード", data=img_bytes, file_name=new_name, mime="image/jpeg")

    # # 円グラフ
    # pie_labels = [result[1] for result in results[:n_top]] + ["others"]
    # pie_probs = [result[2] for result in results[:n_top]] + \
    #             [sum([result[2] for result in results[n_top:]])]

    # fig, ax = plt.subplots()
    # wedgeprops = {"width": 0.3, "edgecolor": "white"}
    # textprops = {"fontsize": 6}
    # ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
    #        textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
    # st.pyplot(fig)
