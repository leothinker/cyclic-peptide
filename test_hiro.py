from hiro_ocsr import HiroOCSR

model = HiroOCSR()
result = model.run(
    {
        "image": "data/processed/cells/PCTCN2025075389-ftappb-I100297_1036_struct.png",
        "depict": False,
    }
)

print(result[0]["SMILES"])
