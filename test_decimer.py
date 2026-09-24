from DECIMER import predict_SMILES

# Unleash the power of DECIMER
image_path = "1038.png"
SMILES = predict_SMILES(image_path)
print(f"🎉 Decoded SMILES: {SMILES}")
