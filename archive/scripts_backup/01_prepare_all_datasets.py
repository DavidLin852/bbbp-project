"""
为所有4个数据集生成特征
"""
import sys
from pathlib import Path
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据集配置
datasets = ['A', 'A_B', 'A_B_C', 'A_B_C_D']

print("="*70)
print("为所有数据集生成特征")
print("="*70)

for dataset in datasets:
    print(f"\n处理数据集: {dataset}")

    # 源数据路径
    source_dir = PROJECT_ROOT / "data" / "splits" / f"seed_0_{dataset}"
    target_dir = PROJECT_ROOT / "data" / "splits" / f"seed_0"

    # 备份原始数据
    if target_dir.exists():
        backup_dir = PROJECT_ROOT / "data" / "splits" / "seed_0_original"
        if not backup_dir.exists():
            shutil.copytree(target_dir, backup_dir)
            print(f"  已备份原始数据到: {backup_dir}")

    # 复制数据集到默认位置
    print(f"  复制 {source_dir} -> {target_dir}")

    # 删除现有数据（如果不是原始备份）
    if target_dir.exists() and target_dir != backup_dir:
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)

    # 生成特征
    print(f"  生成Morgan指纹...")
    from src.featurize.fingerprints import generate_morgan_fingerprints

    try:
        generate_morgan_fingerprints(seed=0)
        print(f"  ✅ 特征生成完成")
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print("\n" + "="*70)
print("完成!")
print("="*70)

# 恢复原始数据（A+B组）
print("\n恢复原始数据集（A+B）...")
backup_dir = PROJECT_ROOT / "data" / "splits" / "seed_0_original"
target_dir = PROJECT_ROOT / "data" / "splits" / f"seed_0"

if backup_dir.exists():
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(backup_dir, target_dir)
    print(f"  已恢复: {target_dir}")
