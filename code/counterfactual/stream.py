#https://qiita.com/sypn/items/80962d84126be4092d3c
# を参考にした

import streamlit as st
from PIL import Image
import cut
import cv2
import numpy as np
import subprocess
import os
import uuid


st.title("顔切り出しアプリ")
# main.pyの引数設定
st.sidebar.header("main.py の引数設定")
target_class = st.sidebar.radio("target_class (0: 女性, 1: 男性)", (0, 1))
save_at = st.sidebar.slider("save_at (分類値)", min_value=0.8, max_value=0.99, value=0.9)
st.write("＞で main.py の引数設定: ", f"--ターゲット {target_class} --save_at {save_at}")
uploaded_file = st.file_uploader("画像をアップロードしてください。", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        cut_image = cut.cut_face(image)
        if cut_image is not None:
            # 一時保存用のファイル名
            temp_filename = f"temp_cut_face_{uuid.uuid4()}.png"
            temp_image_path = os.path.join(os.getcwd(), temp_filename)
            cv2.imwrite(temp_image_path, cut_image)
            # 切り出し画像のリサイズ
            resized_image_path = f"resized_cut_face_{uuid.uuid4()}.png"
            cut.crop_and_resize_image(temp_image_path, resized_image_path)
            st.image(resized_image_path, caption="リサイズされた顔画像", use_column_width=True)

            with st.spinner("処理中..."):
                command = (
                    f"python main.py main data-set --name CelebA classifier --path checkpoints/classifiers/classifier_param.pth "
                    f"generative-model --g_type Flow adv-attack --image_path {resized_image_path} "
                    f"--target_class {target_class} --lr 5e-2 --num_steps 100 --save_at {save_at}"
                )
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            st.subheader("処理結果:")
            if result.returncode == 0:
            # main.py での出力ファイル名に合わせる (UUID部分を除去)
                output_image_path = (f"overview.png")
                try:
                    st.image(output_image_path, caption="生成された画像", use_column_width=True)
                except FileNotFoundError:
                    st.error(f"エラー: 画像ファイル '{output_image_path}' が見つかりません。")
            else:
                st.error(f"エラーが発生しました: \n標準エラー出力:\n{result.stderr}\n標準出力:\n{result.stdout}")

            # 一時ファイルとリサイズ画像の削除
            os.remove(temp_image_path)
            os.remove(resized_image_path)
        else:
            st.write("顔は検出されませんでした。")
    except Exception as e:
        st.error(f"エラーが発生しました: {type(e).__name__}: {e}")

