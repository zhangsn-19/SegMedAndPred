from psd_tools import PSDImage
from PIL import Image

def extract_layer(psd_file, layer_name, background_name, output_file):
    psd = PSDImage.open(psd_file)

    # 查找指定名称的图层
    layer = None
    for child in psd.descendants():
        if child.name == layer_name:
            layer = child
            break
    
    layer_b = None
    for child in psd.descendants():
        if child.name == background_name:
            layer_b = child
            break

    if layer:
        # 提取图层的像素数据
        layer_image = layer.topil()

        # 打开原始图像
        original_image = layer_b.topil()

        # 确保图层图像和原始图像尺寸一致
        if layer_image.size != original_image.size:
            layer_image = layer_image.resize(original_image.size)

        # 创建一个新的图像，将原始图像作为底图，图层图像作为叠加层
        overlay_image = Image.alpha_composite(original_image.convert('RGBA'), layer_image)

        # 保存结果图像
        overlay_image.save(output_file)
        print(f"图层'{layer_name}'已成功叠加到原始图像并保存为'{output_file}'")
    else:
        print(f"未找到名称为'{layer_name}'的图层")

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
    extract_layer(psd_file, layer_name, '背景', 'overlay_psd/' + output_file)
