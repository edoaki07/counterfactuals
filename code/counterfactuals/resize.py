from PIL import Image

def crop_and_resize_image(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size

        # 正方形になるように切り取る範囲を計算
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # 切り取りとリサイズ
        cropped_img = img.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((64, 64))

        # 保存
        resized_img.save(output_path)

# 使い方
image_path = 'images/037774.jpg'  # 処理したい画像のパス
output_path = 'images/image.png'  # 出力画像のパス
crop_and_resize_image(image_path, output_path)
