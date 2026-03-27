"""
下载ChEMBL数据库用于大规模预训练

ChEMBL是最大、最可靠的生物活性分子数据库
包含200万+化合物，涵盖广泛的化学空间
"""
import urllib.request
import gzip
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("ChEMBL数据库下载指南")
print("="*80)

print("""
ChEMBL数据库信息:
- 来源: EMBL-EBI
- 分子数: 200万+ (ChEMBLv33)
- 数据类型: 生物活性分子
- 覆盖范围: 药物、先导化合物、探针分子等
- 优势: 高质量、注释丰富、API稳定

使用方法:
""")

# 方法1: 直接下载链接
chembl_urls = {
    "ChEMBLv33 (SMILES)": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/chembl_33.smi.gz",
    "ChEMBLv33 (完整)": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/chembl_33_sqlite.tar.gz",
}

print("\n方法1: 直接下载（推荐）")
print("-" * 80)
print("使用以下命令手动下载:\n")

for name, url in chembl_urls.items():
    print(f"# {name}")
    print(f"# URL: {url}")
    print(f"wget {url}")
    print(f"# 或")
    print(f"curl -O {url}")
    print()

# 方法2: Python下载（小规模测试）
print("\n方法2: Python下载（前10万分子作为测试）")
print("-" * 80)

try:
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/chembl_33.smi.gz"
    output_file = Path("data/chembl_33.smi.gz")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在下载: {url}")
    print(f"保存到: {output_file}")
    print("\n注意: 文件较大(~200MB)，可能需要几分钟...")

    # 使用tqdm显示进度
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc="下载进度") as t:
        urllib.request.urlretrieve(url,
                                   filename=str(output_file),
                                   reporthook=t.update_to)

    print(f"\n下载完成!")

    # 解压并读取前10万行
    print("\n解压并处理数据...")

    output_dir = Path("data/zinc20")
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = []
    with gzip.open(output_file, 'rt') as f:
        for line in tqdm(f, desc="读取SMILES"):
            if len(smiles_list) >= 100000:  # 限制10万
                break

            parts = line.strip().split('\t')
            if len(parts) >= 2:
                smiles_list.append({
                    'SMILES': parts[0],
                    'CHEMBL_ID': parts[1] if len(parts) > 1 else f"CHEMBL{len(smiles_list)}"
                })

    # 保存
    chembl_file = output_dir / "chembl_100k.csv"
    df = pd.DataFrame(smiles_list)
    df.to_csv(chembl_file, index=False)

    print(f"\n保存 {len(smiles_list):,} 个分子到: {chembl_file}")

    # 显示统计
    print("\n数据统计:")
    print(f"  总分子数: {len(smiles_list):,}")
    print(f"  文件大小: {chembl_file.stat().st_size / 1024 / 1024:.1f} MB")

    # 显示示例
    print("\n示例分子:")
    for i in range(min(5, len(smiles_list))):
        print(f"  {smiles_list[i]['CHEMBL_ID']}: {smiles_list[i]['SMILES'][:60]}...")

    print("\n下一步:")
    print(f"python pretrain_zinc20.py --step download --smiles-file {chembl_file}")

except Exception as e:
    print(f"\n自动下载失败: {e}")
    print("\n请使用手动下载方法:")
    print("\n1. 在浏览器中打开:")
    print("   https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/")
    print("\n2. 下载 chembl_33.smi.gz")
    print("\n3. 解压:")
    print("   gunzip chembl_33.smi.gz")
    print("\n4. 使用:")
    print("   python pretrain_zinc20.py --step download --smiles-file chembl_33.smi")

print("\n" + "="*80)
print("ChEMBL数据特点:")
print("="*80)
print("""
✅ 高质量: 手工注释，可靠性高
✅ 多样性: 涵盖广泛化学空间
✅ 生物活性: 包含药理数据
✅ API稳定: EMBL-EBI维护
✅ 定期更新: 每年发布新版本
✅ 免费开放: 无需注册

适合大规模预训练!""")
