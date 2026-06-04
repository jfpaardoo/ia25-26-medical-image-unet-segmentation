import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    splits_dir = project_root / "data/splits"
    out_dir = project_root / "artifacts/npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        txt_file = splits_dir / f"{split}.txt"
        if not txt_file.exists():
            continue
        
        lines = [line.strip() for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
            
        images = []
        masks = []
        for line in lines:
            img_rel, msk_rel = line.split(",", 1)
            images.append(np.load(project_root / img_rel))
            masks.append(np.load(project_root / msk_rel))
            
        out_file = out_dir / f"{split}.npz"
        np.savez_compressed(out_file, images=np.stack(images), masks=np.stack(masks))
        print(f"Packed {len(images)} patches into {out_file.as_posix()}")

if __name__ == "__main__":
    main()
