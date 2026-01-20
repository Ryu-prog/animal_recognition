import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import io
import zipfile

st.sidebar.title("動物認識アプリ")
st.sidebar.write("生成AIを使って動物園の写真に写っている動物を判定します。")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))

# まず img_file を取得する（ここが最重要）
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
else:
    img_file = st.camera_input("カメラで撮影")

# img_file が None のときは何もしない
if img_file is None:
    st.stop()

# 予測関数をキャッシュ
@st.cache_data
def cached_predict(img_bytes):
    return predict(img_bytes)


# 複数ファイルの場合の処理
if img_source == "画像をアップロード" and isinstance(img_file, list) and len(img_file) > 1:
    # 上限10ファイル
    img_files = img_file[:10] if len(img_file) > 10 else img_file
    
    # 各画像の予測結果を保存
    results_list = []
    images = []
    for i, file in enumerate(img_files):
        img_bytes = file.getvalue()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append((img, file.name.split('.')[-1]))
        
        # キャッシュされた予測
        results = cached_predict(img_bytes)
        results_list.append(results[0])  # トップ結果のみ保存
    
    # 画像を表示（オプション）
    for i, (img, _) in enumerate(images):
        st.image(img, caption=f"対象の画像 {i+1}", width=480)
    
    # ダウンロードボタン（ZIPファイル）
    if len(images) > 0:
        if st.button("画像をダウンロード"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for i, ((img, ext), top_result) in enumerate(zip(images, results_list)):
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='JPEG')
                    img_bytes = img_bytes.getvalue()
                    new_name = f"{i+1}_{top_result[0]}.{ext}"
                    zip_file.writestr(new_name, img_bytes)
            zip_buffer.seek(0)
            st.download_button(label="ZIPファイルをダウンロード", data=zip_buffer, file_name="animal_images.zip", mime="application/zip")

else:
    # 単一ファイルの場合（カメラまたはアップロードの単一）
    with st.spinner("推定中..."):

        # 画像の読み込み
        if img_source == "カメラで撮影":
            img = Image.open(img_file)
            img_bytes = img_file.getvalue()
        else:
            # アップロードの単一ファイル（リストの最初の要素）
            file = img_file[0]
            img_bytes = file.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            ext = file.name.split('.')[-1]

        img = img.convert("RGB")

        st.image(img, caption="対象の画像", width=480)

        # 予測
        results = cached_predict(img_bytes)

        # 結果の表示
        st.subheader("判定結果")
        n_top = 3
        for result in results[:n_top]:
            st.write(f"{round(result[2]*100, 2)}% の確率で {result[0]} です。")

        # 円グラフ
        pie_labels = [result[1] for result in results[:n_top]] + ["others"]
        pie_probs = [result[2] for result in results[:n_top]] + \
                    [sum([result[2] for result in results[n_top:]])]

        fig, ax = plt.subplots()
        wedgeprops = {"width": 0.3, "edgecolor": "white"}
        textprops = {"fontsize": 6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
            textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
        st.pyplot(fig)

        # ダウンロードボタン　- 予測結果がある場合のみ表示
        if results:
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

    
