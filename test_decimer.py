from DECIMER import predict_SMILES

# Unleash the power of DECIMER
image_path = "data/processed/cells/PCTCN2025075389-ftappb-I100297_1036_struct.png"
SMILES = predict_SMILES(image_path)
print(f"🎉 Decoded SMILES: {SMILES}")
