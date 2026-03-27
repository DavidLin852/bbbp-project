"""
真实的ZINC20数据下载方法

ZINC20数据获取途径：
1. 使用zincdownloader Python包
2. 直接从ZINC20 API下载
3. 使用公开的ZINC数据子集
"""
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import io

def method1_zincdownloader():
    """方法1: 使用zincdownloader包"""
    print("\n" + "="*80)
    print("方法1: 使用zincdownloader包")
    print("="*80)

    code = """
# 安装
pip install zincdownloader

# 使用Python代码
from zincdownloader import download

# 下载特定tranche
# ZINC20 tranche IDs: 例如 "2, 3, 5" 代表不同分子量范围
download_zinc20(
    tranches=["2", "3", "5"],  # 小分子tranches
    output_dir="data/zinc20",
    num_molecules=1000000,
    properties=["smiles", "mwt", "logp", "rbonds"]
)
"""
    print(code)
    print("\n注意: 需要先安装 zincdownloader")
    print("pip install zincdownloader")


def method2_zinc20_api_direct():
    """方法2: 直接使用ZINC20 API"""
    print("\n" + "="*80)
    print("方法2: ZINC20 API直接查询")
    print("="*80)

    print("""
ZINC20提供了REST API:

# 获取特定性质范围的分子
https://zinc20.docking.org/substances/subset随机/smiles/?mwt=150-500&logp=-2-5

# 分页下载
https://zinc20.docking.org/substances/subset随机/smiles/?page=1&per_page=1000
    """)

    # 示例：尝试下载前1000个分子
    print("\n尝试下载示例...")
    try:
        url = "https://zinc20.docking.org/substances/subset_random/smiles/?mwt_min=150&mwt_max=500&per_page=1000"
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            print(f"成功下载 {len(lines)} 个分子")
            print(f"前3个示例:")
            for line in lines[:3]:
                parts = line.split()
                if len(parts) >= 2:
                    print(f"  {parts[1]}: {parts[0][:50]}...")
        else:
            print(f"下载失败: {response.status_code}")
    except Exception as e:
        print(f"API请求失败: {e}")


def method3_public_datasets():
    """方法3: 使用公开的ZINC数据集"""
    print("\n" + "="*80)
    print("方法3: 公开数据集源")
    print("="*80)

    datasets = {
        "ZINC12": "http://zinc.docking.org/db/bysubset/drug/like/1/1/1_smi.ows",
        "ZINC15 (Lead-like)": "http://zinc15.docking.org/subsets/lead-like/tracts/1/1.smi",
        "ZINC15 (Drug-like)": "http://zinc15.docking.org/subsets/drug-like/tracts/1/1.smi",
        "ZINC15 (Fragment)": "http://zinc15.docking.org/subsets/fragment/tracts/1/1.smi",
    }

    print("\n可用的公开数据集:")
    for name, url in datasets.items():
        print(f"\n{name}:")
        print(f"  URL: {url}")

    # 尝试下载ZINC12（最稳定）
    print("\n" + "-"*80)
    print("尝试下载ZINC12 (drug-like subset)...")
    try:
        url = "http://zinc.docking.org/db/bysubset/drug/like/1/1/1_smi.ows"
        response = requests.get(url, stream=True, timeout=120)

        if response.status_code == 200:
            smiles_list = []
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        smiles_list.append({'SMILES': parts[0], 'ZINC_ID': parts[1]})

                if len(smiles_list) >= 10000:  # 限制1万作为示例
                    break

            print(f"成功下载 {len(smiles_list):,} 个分子")

            # 保存
            output_dir = Path("data/zinc20")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "zinc12_druglike_10k.csv"

            df = pd.DataFrame(smiles_list)
            df.to_csv(output_file, index=False)
            print(f"保存到: {output_file}")

            return output_file

    except Exception as e:
        print(f"ZINC12下载失败: {e}")

    return None


def method4_generate_diverse():
    """方法4: 从SMILES库生成多样性分子"""
    print("\n" + "="*80)
    print("方法4: 从现有SMILES库生成")
    print("="*80)

    print("""
如果你有其他SMILES文件（如ChEMBL, PubChem等），可以使用：

# 从ChEMBL下载
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLv33/chembl_33.smi.gz

# 从PubChem下载
https://pubchem.ncbi.nlm.nih.gov/source/

# 然后运行预训练
python pretrain_zinc20.py --step download --smiles-file your_data.csv
    """)


def analyze_current_data():
    """分析当前使用的数据"""
    print("\n" + "="*80)
    print("当前预训练数据来源分析")
    print("="*80)

    print("""
📊 当前数据 (4,611 分子):
   来源: BBB数据集 (blood-brain barrier permeability)
   特点: 100% 药物分子 (符合Lipinski规则)
   问题: 分子多样性有限，都是潜在的BBB渗透分子

⚠️  这不是真正的ZINC20数据！

✅ 真正的ZINC20应该包含：
   - 10亿+ 可商业化分子
   - 高度多样性（药物、先导化合物、片段等）
   - 更广泛的化学空间
   - 不限于药物分子
    """)


def main():
    print("="*80)
    print("真实的ZINC20数据下载指南")
    print("="*80)

    analyze_current_data()

    print("\n选择数据获取方法:")
    print("1. zincdownloader包 (推荐)")
    print("2. ZINC20 API")
    print("3. 公开数据集")
    print("4. 其他来源")

    # 尝试方法3（最可靠）
    result = method3_public_datasets()

    if result:
        print(f"\n✅ 成功获取真实数据!")
        print(f"可以使用: python pretrain_zinc20.py --step download --smiles-file {result}")

    print("\n" + "="*80)
    print("总结建议")
    print("="*80)
    print("""
对于大规模预训练，推荐：

1️⃣ 小规模测试 (1-10万分子):
   使用ZINC12 drug-like subset
   或从ChEMBL/PubChem采样

2️⃣ 中规模 (10-100万分子):
   组合多个ZINC15 subset

3️⃣ 大规模 (100万+分子):
   安装zincdownloader包
   或直接从ZINC20 FTP下载

当前BBB数据（4,611分子）:
   ✅ 适合快速测试
   ❌ 不代表真实化学空间多样性
   ⚠️  过拟合风险
    """)


if __name__ == "__main__":
    main()
