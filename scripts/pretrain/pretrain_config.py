"""
Pretraining Configuration for CFFF Cluster

使用此配置文件快速设置预训练参数
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusterConfig:
    """集群配置"""
    project_dir: str = "/path/to/your/bbbp-project"  # 需要修改
    data_dir: str = None  # 自动设置为 project_dir/data/zinc22
    conda_env: str = "bbb"
    partition: str = "gpu"
    cpus_per_task: int = 8
    mem_gb: int = 32
    time_hours: int = 24

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = str(Path(self.project_dir) / "data" / "zinc22")


@dataclass
class GraphPretrainConfig:
    """图预训练配置"""
    num_samples: int = 1000000  # 1M samples
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3

    # 模型架构
    model_type: str = "gin"  # or "gat"
    hidden_dim: int = 256
    num_layers: int = 5

    # 其他
    device: str = "auto"
    save_dir: str = None  # 自动设置

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = "artifacts/models/pretrain/graph"


@dataclass
class TransformerPretrainConfig:
    """Transformer预训练配置"""
    num_samples: int = 1000000  # 1M samples
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    mask_ratio: float = 0.15

    # 模型架构
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6

    # 其他
    device: str = "auto"
    save_dir: str = None  # 自动设置

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = "artifacts/models/pretrain/transformer"


@dataclass
class TestConfig:
    """快速测试配置"""
    num_samples: int = 10000
    epochs: int = 5
    batch_size: int = 32


# 预定义配置方案
class Presets:
    """预定义配置方案"""

    class QuickTest:
        """快速测试（30分钟）"""
        cluster = ClusterConfig(time_hours=2, mem_gb=16)
        graph = GraphPretrainConfig(
            num_samples=10000,
            epochs=5,
            batch_size=32,
            hidden_dim=128,
            num_layers=3,
        )
        transformer = TransformerPretrainConfig(
            num_samples=10000,
            epochs=5,
            batch_size=32,
            d_model=256,
            n_layers=4,
        )

    class Standard:
        """标准训练（24小时每个）"""
        cluster = ClusterConfig(time_hours=24)
        graph = GraphPretrainConfig()  # 使用默认值
        transformer = TransformerPretrainConfig()  # 使用默认值

    class LargeModel:
        """大模型训练（更多GPU资源）"""
        cluster = ClusterConfig(
            cpus_per_task=16,
            mem_gb=64,
            time_hours=48
        )
        graph = GraphPretrainConfig(
            batch_size=256,
            hidden_dim=512,
            num_layers=8,
        )
        transformer = TransformerPretrainConfig(
            batch_size=256,
            d_model=1024,
            n_heads=16,
            n_layers=12,
        )

    class LimitedGPU:
        """GPU内存受限"""
        cluster = ClusterConfig(mem_gb=16)
        graph = GraphPretrainConfig(
            batch_size=64,
            hidden_dim=128,
            num_layers=3,
        )
        transformer = TransformerPretrainConfig(
            batch_size=64,
            d_model=256,
            n_heads=8,
            n_layers=4,
        )


def generate_slurm_script(
    config: ClusterConfig,
    script_name: str,
    commands: list[str],
) -> str:
    """
    生成SLURM提交脚本

    Args:
        config: 集群配置
        script_name: 脚本名称
        commands: 要运行的命令列表

    Returns:
        脚本内容
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={script_name}
#SBATCH --output=logs/{script_name}_%j.out
#SBATCH --error=logs/{script_name}_%j.err
#SBATCH --time={config.time_hours}:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem_gb}G
#SBATCH --partition={config.partition}
#SBATCH --gres=gpu:1

set -e

echo "=================================================="
echo "{script_name} - Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

cd {config.project_dir}
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate {config.conda_env}

# Check GPU
nvidia-smi

echo ""
echo "Running commands..."
echo "=================================================="

"""

    for cmd in commands:
        script += f"\n{cmd}\n"

    script += f"""
echo ""
echo "=================================================="
echo "Completed at: $(date)"
echo "=================================================="
"""
    return script


# 使用示例
if __name__ == "__main__":
    # 选择预设
    preset = Presets.Standard

    # 生成图预训练脚本
    graph_commands = [
        f"python scripts/pretrain/pretrain_graph.py \\",
        f"    --data_dir {preset.cluster.data_dir} \\",
        f"    --num_samples {preset.graph.num_samples} \\",
        f"    --batch_size {preset.graph.batch_size} \\",
        f"    --epochs {preset.graph.epochs} \\",
        f"    --model_type {preset.graph.model_type} \\",
        f"    --hidden_dim {preset.graph.hidden_dim} \\",
        f"    --num_layers {preset.graph.num_layers} \\",
        f"    --lr {preset.graph.lr} \\",
        f"    --save_dir {preset.graph.save_dir} \\",
        f"    --device {preset.graph.device}",
    ]

    script = generate_slurm_script(
        preset.cluster,
        "graph_pretrain",
        graph_commands,
    )

    print("Generated SLURM script:")
    print(script)
