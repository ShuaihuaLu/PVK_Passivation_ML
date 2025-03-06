from rdkit import Chem
from rdkit.Chem import AllChem

# 定义SMILES字符串
smiles = 'COC1=CC2=C(C=C1)N(C3=C2C=C(C=C3)OC)CCCCP(=O)(O)O'

# 从SMILES创建分子对象
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("无法解析SMILES字符串")

# 添加氢原子
mol = Chem.AddHs(mol)

# 生成三维构象
status = AllChem.EmbedMolecule(mol)
if status == -1:
    raise RuntimeError("生成三维构象失败")

# MMFF力场优化
AllChem.MMFFOptimizeMolecule(mol)

# 保存为SDF文件
writer = Chem.SDWriter('molecule_3d.sdf')
writer.write(mol)
writer.close()

print("SDF文件已成功保存为 molecule_3d.sdf")
