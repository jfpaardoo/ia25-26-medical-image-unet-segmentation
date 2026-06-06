import numpy as np
from pathlib import Path

def _parse_lines(txt_file: Path) -> dict[str, list[str]]:
    lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    img_to_masks = {}
    for line in lines:
        img_rel, msk_rel = line.split(",", 1)
        if img_rel not in img_to_masks:
            img_to_masks[img_rel] = []
        img_to_masks[img_rel].append(msk_rel)
    return img_to_masks

def _pack_split(txt_file: Path, out_file: Path, project_root: Path) -> None:
    if not txt_file.exists():
        return
        
    img_to_masks = _parse_lines(txt_file)
    if not img_to_masks:
        return

    images = []
    masks_1 = []
    masks_2 = []
    groups = []
    
    for img_rel, msk_rels in img_to_masks.items():
        images.append(np.load(project_root / img_rel))
        msk_rels.sort()
        masks_1.append(np.load(project_root / msk_rels[0]))
        if len(msk_rels) > 1:
            masks_2.append(np.load(project_root / msk_rels[1]))
            
        basename = Path(img_rel).stem
        parts = basename.split("__")
        groups.append("__".join(parts[:-2]))
        
    out_dict = {"images": np.stack(images)}
    if masks_2:
        out_dict["masks_expert1"] = np.stack(masks_1)
        out_dict["masks_expert2"] = np.stack(masks_2)
    else:
        out_dict["masks"] = np.stack(masks_1)
        
    if groups:
        out_dict["groups"] = np.array(groups, dtype=str)
        
    np.savez_compressed(out_file, **out_dict)
    print(f"Packed {len(images)} patches into {out_file.as_posix()}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    splits_dir = project_root / "data/splits"
    out_dir = project_root / "artifacts/npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        txt_file = splits_dir / f"{split}.txt"
        out_file = out_dir / f"{split}.npz"
        _pack_split(txt_file, out_file, project_root)

if __name__ == "__main__":
    main()
