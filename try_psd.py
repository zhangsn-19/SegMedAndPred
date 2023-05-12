# https://psd-tools.readthedocs.io/en/latest/

from psd_tools import PSDImage

psd = PSDImage.open("dir_examples/10109280 前胸 注意皮靴底儿的皮纹拉长线 (2)  2022-8-22.psd")
psd.composite().save("gen_psd/gen1.png")
for cnt, layer in enumerate(psd):
    print(layer)
    image = layer.composite()
    image.save("gen_psd/gen" + str(cnt) + ".png")

