import os
import glob
import torch

base = "artifacts/models/pretrain/exp_matrix"

for exp_dir in sorted(os.listdir(base)):
    full_dir = os.path.join(base, exp_dir)
    if not os.path.isdir(full_dir):
        continue

    # Find epoch checkpoint files (exclude backbone files which have no history)
    ckpts = (
        glob.glob(os.path.join(full_dir, "*pretrain_epoch*.pt")) +
        glob.glob(os.path.join(full_dir, "*denoise_*_epoch*.pt")) +
        glob.glob(os.path.join(full_dir, "*transformer_pretrain_epoch*.pt"))
    )
    if not ckpts:
        # fallback: any epoch checkpoint
        ckpts = glob.glob(os.path.join(full_dir, "*epoch*.pt"))

    if not ckpts:
        print(f"{exp_dir}: NO CHECKPOINT FOUND")
        continue

    # Load the LAST epoch checkpoint (highest epoch number in filename)
    def epoch_from_path(p):
        import re
        m = re.search(r'epoch[_\-]?(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpts.sort(key=epoch_from_path, reverse=True)
    ckpt_file = ckpts[0]

    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{exp_dir}: LOAD ERROR ({e})")
        continue

    h = ckpt.get("history", {})
    losses = h.get("train_loss", [])

    if losses:
        print(f"{exp_dir}: epochs={len(losses)}, first={losses[0]:.6f}, last={losses[-1]:.6f}")
    else:
        print(f"{exp_dir}: NO LOSS HISTORY (file: {os.path.basename(ckpt_file)})")
