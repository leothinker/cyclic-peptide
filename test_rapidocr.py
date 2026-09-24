from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "data/raw/WO2025162428/golden_tables/PCTCN2025075389-ftappb-I100297.jpg"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
