#!/usr/bin/env python3
# homework/auto_generate_captions.py

import json
from pathlib import Path
import fire

from homework.generate_captions import generate_caption


def process_dir(data_dir: str, views: int = -1):
    """
    Process all *_info.json inside data/train or data/valid.
    Produces <scene>_captions.json containing:
      { "image_file": "...", "captions": [...] }
    """
    data_dir = Path(data_dir)
    info_files = sorted(data_dir.glob("*_info.json"))

    print(f"Found {len(info_files)} info files in {data_dir}")

    for info_path in info_files:
        base = info_path.stem.replace("_info", "")
        out_entries = []

        # Find all view indexes by scanning filenames
        if views == -1:
            view_indexes = sorted({int(p.name.split("_")[1]) for p in data_dir.glob(f"{base}_*_im.*")})
        else:
            view_indexes = list(range(views))

        for view in view_indexes:
            caps = generate_caption(str(info_path), view)

            img_candidates = sorted(info_path.parent.glob(f"{base}_{view:02d}_im.*"))
            if not img_candidates:
                continue

            out_entries.append({
                "image_file": str(img_candidates[0].relative_to(data_dir.parent)),
                "captions": caps,
            })

        out_file = info_path.parent / f"{base}_captions.json"
        out_file.write_text(json.dumps(out_entries, indent=2))

        print("Wrote:", out_file)


def main():
    fire.Fire({"run": process_dir})


if __name__ == "__main__":
    main()
