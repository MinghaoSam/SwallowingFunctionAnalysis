from PIL import Image
import os

path = './annotated/my_markEndpoints'

# 获取文件夹内的图像文件列表
image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

# 获取前八张图片
image_files = image_files[:8]

# 打开图像文件并调整尺寸
images = []
for file in image_files:
    image = Image.open(os.path.join(path, file))
    image.thumbnail((700, 700), Image.LANCZOS)  # 调整尺寸并保持宽高比
    images.append(image)

# 保存为GIF图像
output_file = 'swallowing_markEndpoints.gif'
images[0].save(output_file, save_all=True, append_images=images[1:], duration=200, loop=0)

