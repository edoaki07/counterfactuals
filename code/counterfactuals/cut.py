import cv2
from PIL import Image

# https://qiita.com/stpete_ishii/items/c4030ab664fab23313b0
# を参考にしました
#学習済みモデル https://github.com/opencv/opencv/tree/master/data/haarcascades

def cut_face(image):
    """
    画像ファイルパスを受け取り、顔部分を切り出す関数

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        numpy.ndarray or None: 切り出された顔画像 (NumPy 配列)、顔が検出されなかった場合は None
    """

    if image is None:  # 画像の読み込みに失敗した場合
        return None

    # グレースケール変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # カスケード分類器の読み込み
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    # 顔部分が抽出できたならサイズを変更しface_imageに格納
    if len(face_list) > 0:
        for i, rect in enumerate(face_list):
            x, y, width, height = rect

            # 拡張率を指定 (例: 20%)
            expand_ratio_x = 0.4  # 横方向の拡張率
            expand_ratio_y = 0.4  # 縦方向の拡張率
            top_shift_ratio = 0.25  # 上方向への寄せの割合

            # 上方向へのシフト量を計算
            top_shift = int(height * top_shift_ratio)

            # 拡張後の座標とサイズを計算 (上方向に寄せ、全体を少し小さく)
            x = max(0, int(x - width * expand_ratio_x / 2))  # 横方向は中央寄せ
            y = max(0, int(y - top_shift))  # 上方向に寄せる
            width = int(width * (1 + expand_ratio_x) * 0.95)  # 横幅を少し小さく
            height = int(height * (1 + expand_ratio_y) * 0.95)  # 高さを少し小さく

            # 画像の顔部分のみを切り出す
            image = image[y:y + height, x:x + width]

            if image.shape[0] < 64:
                continue

            image = cv2.resize(image, (512, 512))
            return image
    else:
        print("顔は検出されませんでした。")
        return image

def crop_and_resize_image(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        cropped_img = img.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((64, 64))
        resized_img.save(output_path)
