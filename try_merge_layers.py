from psd_tools import PSDImage
from PIL import Image
import numpy as np
import cv2
import ipdb

def merge_layers_another(psd_file, layer1_name, layer2_name, output_file):
    psd = PSDImage.open(psd_file)

    # Find the first layer
    layer1 = None
    for layer in psd.descendants():
        if layer.name == layer1_name:
            layer1 = layer
            break

    # Find the second layer
    layer2 = None
    for layer in psd.descendants():
        if layer.name == layer2_name:
            layer2 = layer
            break

    if layer1 and layer2:
        # Merge the layers
        merged_image = Image.alpha_composite(layer1.topil(), layer2.topil())

        # Save the merged image as PNG
        merged_image.save(output_file, format='PNG')
        print(f"Layers '{layer1_name}' and '{layer2_name}' have been successfully merged and saved as '{output_file}'")
    else:
        if not layer1:
            print(f"Layer '{layer1_name}' not found")
        if not layer2:
            print(f"Layer '{layer2_name}' not found")

def extract_arrow_direction(psd_file, arrow_layer_name, arrow_png_file):
    # Load the PSD image
    psd = PSDImage.open(psd_file)

    # Find the arrow layer
    arrow_layer = None
    for layer in psd.descendants():
        if layer.name == arrow_layer_name:
            arrow_layer = layer
            break

    if arrow_layer:
        # Convert the arrow layer to a binary mask
        arrow_mask = arrow_layer.topil()

        # Load the arrow PNG image
        # arrow_image = cv2.imread(arrow_png_file, cv2.IMREAD_GRAYSCALE)

        # Apply morphological operations to enhance the arrow shape
        kernel = np.ones((5, 5), np.uint8)
        arrow_mask = np.array(arrow_mask)
        arrow_image = cv2.morphologyEx(arrow_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the arrow image
        # new_arrow_image = cv2.cvtColor(arrow_image, cv2.COLOR_RGBA2GRAY)
        # ipdb.set_trace()
        packed = cv2.findContours(arrow_image[:,:,3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = packed[0][0]
        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a line to the largest contour
            vx, vy, _, _ = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate the angle of the line
            angle = np.arctan2(vy, vx) * 180 / np.pi
            if angle < 0:
                angle += 360

            print(angle)

            print(f"The arrow direction angle is: {angle} degrees")
        else:
            print("No arrow contour found in the PNG image")
    else:
        print(f"Arrow layer '{arrow_layer_name}' not found in the PSD file")

def merge_layers_new(psd_file, layer1_name, layer2_name, output_file):
    psd = PSDImage.open(psd_file)

    # 查找第一个图层
    layer1 = None
    for child in psd.descendants():
        if child.name == layer1_name:
            layer1 = child
            break

    # 查找第二个图层
    layer2 = None
    for child in psd.descendants():
        if child.name == layer2_name:
            layer2 = child
            break

    if layer1 and layer2:
        # 获取原始图像尺寸
        width, height = psd.size

        # 创建空白图像，与原始图像尺寸相同
        merged_image = Image.new("RGBA", (width, height))

        # 获取图层1的位置
        layer1_left = layer1.left
        layer1_top = layer1.top

        # 获取图层2的位置
        layer2_left = layer2.left
        layer2_top = layer2.top

        # 将图层1绘制到合并图像上
        # 将图层2绘制到合并图像上
        merged_image.paste(layer2.topil(), (layer2_left, layer2_top))
        merged_image.paste(layer1.topil(), (layer1_left, layer1_top))

        # 保存合并后的图像为PNG文件
        merged_image.save(output_file, format='PNG')
        print(f"图层'{layer1_name}'和'{layer2_name}'已成功合并并保存为'{output_file}'")
    else:
        if not layer1:
            print(f"未找到名称为'{layer1_name}'的图层")
        if not layer2:
            print(f"未找到名称为'{layer2_name}'的图层")

def merge_layers(psd_file, layer1_name, layer2_name, output_file):
    psd = PSDImage.open(psd_file)

    # 查找第一个图层
    layer1 = None
    for child in psd.descendants():
        if child.name == layer1_name:
            layer1 = child
            break

    # 查找第二个图层
    layer2 = None
    for child in psd.descendants():
        if child.name == layer2_name:
            layer2 = child
            break

    if layer1 and layer2:
        # 获取原始图像尺寸
        width, height = psd.header.width, psd.header.height

        # 创建空白图像，与原始图像尺寸相同
        merged_image = Image.new("RGBA", (width, height))

        # 获取图层1的位置
        layer1_left = layer1.left
        layer1_top = layer1.top

        # 获取图层2的位置
        layer2_left = layer2.left
        layer2_top = layer2.top

        # 将图层1绘制到合并图像上
        merged_image.paste(layer1.topil(), (layer1_left, layer1_top))

        # 将图层2绘制到合并图像上
        merged_image.paste(layer2.topil(), (layer2_left, layer2_top), layer2.topil())

        # 保存合并后的图像为PNG文件
        merged_image.save(output_file, format='PNG')
        print(f"图层'{layer1_name}'和'{layer2_name}'已成功合并并保存为'{output_file}'")
    else:
        if not layer1:
            print(f"未找到名称为'{layer1_name}'的图层")
        if not layer2:
            print(f"未找到名称为'{layer2_name}'的图层")

# 调用示例
psd_file_list = ['dir_examples/10109280 前胸 注意皮靴底儿的皮纹拉长线 (2)  2022-8-22.psd',
                'dir_examples/10110505 前胸2 (1) 2021-11-22.psd',
                'dir_examples/10155290 前胸 1,  (1) 2022-5-5.psd',
                'dir_examples/10307034  前胸 1 (2) 2021-12-16.psd']
# 输入PSD文件路径
layer_names_list = [
    '形状 1', '形状 1', '形状 1', '形状 1'
]  # 要提取的图层名称
output_file_list = [
    'gen1.png',
    'gen2.png',
    'gen3.png',
    'gen4.png'
]  # 输出提取图层的PNG文件路径

for (psd_file, layer_name, output_file) in zip(psd_file_list, layer_names_list, output_file_list):
    # merge_layers_new(psd_file, layer_name, '背景', 'overlay_psd/' + output_file)
    # psd_file, arrow_layer_name, arrow_png_file
    extract_arrow_direction(psd_file, layer_name, 'overlay/' + output_file)
