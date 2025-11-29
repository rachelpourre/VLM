#!/usr/bin/env python3
# auto_generate_captions.py

import json
from pathlib import Path
import fire

from homework.generate_captions import generate_caption


def process_dir(data_dir: str, output_dir: str, views=3):
    """
    Automatically generate captions for all *_info.json files.

    Writes:
       data/train/00000_captions.json
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info_files = sorted(data_dir.glob("*_info.json"))

    print(f"Found {len(info_files)} info files in {data_dir}")

    for info_path in info_files:
        base = info_path.stem.replace("_info", "")

        all_entries = []

        for view in range(views):
            captions = generate_caption(str(info_path), view)

            # Find image file
            candidates = list(info_path.parent.glob(f"{base}_{view:02d}_im.jpg"))
            if not candidates:
                continue
            image_path = str(candidates[0])

            all_entries.append({
                "image": image_path,
                "captions": captions,
            })

        out_file = output_dir / f"{base}_captions.json"
        with open(out_file, "w") as f:
            json.dump(all_entries, f, indent=2)

        print("Wrote:", out_file)


def main():
    fire.Fire({
        "run": process_dir,
    })


if __name__ == "__main__":
    main()
