from DECIMER import predict_SMILES

# Unleash the power of DECIMER
image_path = "data/processed/cells/PCTCN2025075389-ftappb-I100297_1036_struct.png"
SMILES = predict_SMILES(image_path)
print(f"🎉 Decoded SMILES: {SMILES}")
# CC[C@H](C)[C@H]1C(=O)N(C)[C@H](C2CCCCC2)C(=O)N3CC[C@@H]3C(=O)N[C@@H]4CSC5=C(C(=C(C(=C5F)F)C(=S)NCCC(=O)N1)F)SC[C@@H](C(=O)N(C)CC(=O)N[C@H](CCC6=CC(=C(C(=C6)F)C(F)(F)F)F)C(=O)N7CCC[C@@H]7C(=O)NC8(CCCC8)C(=O)N(C)[C@@H](C9CCCC9)C(=O)N(C)[C@@H](CC(=O)N(C)C)C(=O)N(C)C)NC4=O
