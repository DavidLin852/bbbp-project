import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw


def draw_smarts_from_json(
    json_path,
    out_dir="artifacts/smarts_viz",
    mols_per_row=4,
    sub_img_size=(300, 300)
):
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mols = []
    legends = []
    failed = []

    for entry in data:
        name = entry.get("name", "UNKNOWN")
        smarts = entry.get("smarts", "")

        try:
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                print(f"[WARN] failed to parse: {name} -> {smarts}")
                failed.append((name, smarts))
                continue
            mols.append(mol)
            legends.append(name)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed.append((name, smarts))

    print(f"[INFO] total SMARTS: {len(data)}")
    print(f"[INFO] parsed OK: {len(mols)}")
    print(f"[INFO] failed: {len(failed)}")

    # =========================
    # 1. 生成一张总览网格图
    # =========================
    grid_img = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        useSVG=False
    )

    grid_path = out_dir / "all_smarts_grid.png"
    grid_img.save(str(grid_path))
    print(f"[OK] grid saved to: {grid_path}")

    # =========================
    # 2. 每个 SMARTS 单独一张
    # =========================
    single_dir = out_dir / "single"
    single_dir.mkdir(exist_ok=True)

    for mol, name in zip(mols, legends):
        img = Draw.MolToImage(mol, size=sub_img_size)
        img_path = single_dir / f"{name}.png"
        img.save(str(img_path))

    print(f"[OK] single images saved to: {single_dir}")

    # =========================
    # 3. 保存失败列表，方便你排查
    # =========================
    if failed:
        fail_path = out_dir / "failed_smarts.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for name, smarts in failed:
                f.write(f"{name}\t{smarts}\n")
        print(f"[WARN] failed SMARTS written to: {fail_path}")


if __name__ == "__main__":
    # 👉 改成你真实的路径
    json_path = "data/smarts_definitions.json"

    draw_smarts_from_json(
        json_path=json_path,
        out_dir="artifacts/smarts_viz",
        mols_per_row=4,
        sub_img_size=(300, 300)
    )
