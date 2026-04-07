#!/usr/bin/env python
"""
Pretraining Readiness Check

检查是否准备好在集群上运行预训练
"""

import sys
from pathlib import Path
from typing import List, Tuple


def check_dependencies() -> Tuple[bool, List[str]]:
    """检查Python依赖"""
    print("检查Python依赖...")
    missing = []

    required = {
        'torch', 'torch_geometric', 'rdkit', 'transformers',
        'numpy', 'pandas', 'tqdm', 'scikit-learn'
    }

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (缺失)")
            missing.append(pkg)

    return len(missing) == 0, missing


def check_data_structure(data_dir: Path) -> Tuple[bool, List[str]]:
    """检查ZINC22数据目录结构"""
    print(f"\n检查ZINC22数据目录: {data_dir}")
    issues = []

    if not data_dir.exists():
        print(f"  ✗ 数据目录不存在")
        return False, ["数据目录不存在"]

    # 检查子目录
    subdirs = list(data_dir.glob("H*"))
    if len(subdirs) == 0:
        print(f"  ✗ 未找到H*子目录 (H04, H05, etc.)")
        issues.append("缺少H*子目录")
    else:
        print(f"  ✓ 找到 {len(subdirs)} 个子目录")

    # 检查.smi.gz文件
    smi_files = list(data_dir.rglob("*.smi.gz"))
    if len(smi_files) == 0:
        print(f"  ✗ 未找到.smi.gz文件")
        issues.append("缺少.smi.gz文件")
    else:
        print(f"  ✓ 找到 {len(smi_files)} 个.smi.gz文件")

    return len(issues) == 0, issues


def check_scripts() -> Tuple[bool, List[str]]:
    """检查训练脚本"""
    print("\n检查训练脚本...")
    missing = []

    scripts = [
        "scripts/pretrain/pretrain_graph.py",
        "scripts/pretrain/pretrain_smiles.py",
    ]

    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (缺失)")
            missing.append(script)

    return len(missing) == 0, missing


def check_gpu():
    """检查GPU可用性"""
    print("\n检查GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ 检测到 {gpu_count} 个GPU")
            print(f"    GPU 0: {gpu_name}")
            return True
        else:
            print(f"  ✗ 未检测到CUDA")
            return False
    except:
        print(f"  ? 无法检查GPU")
        return None


def main():
    print("=" * 60)
    print("B3P 预训练就绪检查")
    print("=" * 60)

    all_good = True

    # 1. 检查依赖
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_good = False
        print(f"\n安装缺失的依赖:")
        print(f"  pip install {' '.join(missing_deps)}")

    # 2. 检查数据
    data_dir = Path("data/zinc22")
    data_ok, data_issues = check_data_structure(data_dir)
    if not data_ok:
        all_good = False
        print(f"\n数据问题:")
        for issue in data_issues:
            print(f"  - {issue}")

    # 3. 检查脚本
    scripts_ok, missing_scripts = check_scripts()
    if not scripts_ok:
        all_good = False
        print(f"\n缺失的脚本:")
        for script in missing_scripts:
            print(f"  - {script}")

    # 4. 检查GPU
    gpu_available = check_gpu()

    # 总结
    print("\n" + "=" * 60)
    if all_good:
        print("✓ 系统就绪！可以开始预训练")
        print("\n推荐命令:")
        print("  本地测试: python scripts/pretrain/pretrain_graph.py --num_samples 10000 --epochs 5")
        print("  集群训练: sbatch scripts/pretrain/run_graph_pretrain_cluster.sh")
    else:
        print("✗ 系统未就绪，请解决上述问题")

    if gpu_available is False:
        print("\n警告: 未检测到GPU，预训练将非常慢")
        print("建议使用GPU集群进行预训练")

    print("=" * 60)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
