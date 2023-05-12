from psd_tools import PSDImage
from PIL import Image

def extract_layer(psd_file, layer_name, output_file):
    psd = PSDImage.open(psd_file)
    
    # 查找指定名称的图层
    layer = None
    for child in psd.descendants():
        if child.name == layer_name:
            layer = child
            break
    
    if layer:
        # 提取图层的像素数据
        layer_image = layer.topil()
        
        # 保存图层为PNG文件
        layer_image.save(output_file, format='PNG')
        print(f"图层'{layer_name}'已成功提取并保存为'{output_file}'")
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
    'gen_psd/gen1.png',
    'gen_psd/gen2.png',
    'gen_psd/gen3.png',
    'gen_psd/gen4.png'
]  # 输出提取图层的PNG文件路径

for (psd_file, layer_name, output_file) in zip(psd_file_list, layer_names_list, output_file_list):
    extract_layer(psd_file, layer_name, output_file)
