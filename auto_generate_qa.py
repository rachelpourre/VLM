# homework/auto_generate_qa.py
import json
from pathlib import Path
import random
import fire

from homework.generate_qa import generate_qa_pairs

# Defaults: these match your repo layout: data/train/*.jpg and *_info.json
DEFAULT_DATA_DIR = "data"
DEFAULT_TRAIN_SUBDIR = "train"

def _frame_id_from_scene(scene_name: str) -> str:
    """
    scene_name is the stem like '00048' or '00048_info' cleaned outside.
    We will return it as-is (preserves hex).
    """
    return scene_name

def process_info_json(
    info_path: str,
    data_dir: str = DEFAULT_DATA_DIR,
    train_subdir: str = DEFAULT_TRAIN_SUBDIR,
    max_relations: int = 4,
    views: int = -1,          # -1 means all views; 1 means view 0 only
    sample_seed: int = 17,
    max_qas_per_image: int = None  # optional cap
):
    """
    Process a single *_info.json file, generate QA pairs for each view/image,
    and write <scene>_qa_pairs.json into the same train folder.
    QA entries include "image_file" as a relative path starting with "train/..."
    """
    DATA_DIR = Path(data_dir)
    TRAIN_DIR = DATA_DIR / train_subdir

    info_path = Path(info_path)
    if not info_path.exists():
        print("Info not found:", info_path)
        return

    scene_id = info_path.stem.replace("_info", "")
    all_images = sorted(TRAIN_DIR.glob(f"{scene_id}_*_im.*"))

    if not all_images:
        print("No images found for", scene_id)
        return

    # detect number of views from filenames (by splitting after scene_id)
    # But simpler: if views == -1, process all images; else only views count (0..views-1)
    if views is None or int(views) < 0:
        process_view_indexes = sorted({int(p.name.split("_")[1]) for p in all_images})
    else:
        # user supplied number of views (e.g., 1 => only view 0)
        v = int(views)
        if v <= 0:
            process_view_indexes = sorted({int(p.name.split("_")[1]) for p in all_images})
        else:
            process_view_indexes = list(range(v))

    rng = random.Random(sample_seed)

    all_qas_out = []

    for view_idx in process_view_indexes:
        # find image file for this view (take first matching extension)
        candidates = sorted(TRAIN_DIR.glob(f"{scene_id}_{view_idx:02d}_im.*"))
        if not candidates:
            # no image for this view
            continue
        img_path = candidates[0]
        rel_img_path = str(img_path.relative_to(DATA_DIR))  # yields "train/00048_00_im.jpg"

        # call generate_qa_pairs - it expects info_path and view_index
        qas = generate_qa_pairs(str(info_path), view_idx, max_relations=max_relations, sample_seed=sample_seed)

        # optionally cap number of qa items per image
        if max_qas_per_image is not None and len(qas) > max_qas_per_image:
            # deterministic cap
            qas = qas[:max_qas_per_image]

        # attach image_file and push out
        for qa in qas:
            qa_out = {
                "image_file": rel_img_path,
                "question": qa["question"],
                "answer": qa["answer"]
            }
            all_qas_out.append(qa_out)

    # write output to TRAIN_DIR/<scene>_qa_pairs.json
    out_file = TRAIN_DIR / f"{scene_id}_qa_pairs.json"
    with open(out_file, "w") as f:
        json.dump(all_qas_out, f, indent=2)

    print(f"Wrote {out_file} ({len(all_qas_out)} QA pairs)")

def run(
    data_dir: str = DEFAULT_DATA_DIR,
    train_subdir: str = DEFAULT_TRAIN_SUBDIR,
    max_relations: int = 4,
    views: int = -1,
    sample_seed: int = 17,
    max_qas_per_image: int = None
):
    """
    Process all *_info.json in data/train and generate corresponding QA pair files.
    Example:
      python -m homework.auto_generate_qa run --data_dir data --train_subdir train --max_relations 4 --views 1
    """
    DATA_DIR = Path(data_dir)
    TRAIN_DIR = DATA_DIR / train_subdir

    info_files = sorted(TRAIN_DIR.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {TRAIN_DIR}")

    for info_path in info_files:
        process_info_json(
            str(info_path),
            data_dir=data_dir,
            train_subdir=train_subdir,
            max_relations=max_relations,
            views=views,
            sample_seed=sample_seed,
            max_qas_per_image=max_qas_per_image
        )

def main():
    fire.Fire({
        "process": process_info_json,
        "run": run
    })

if __name__ == "__main__":
    main()
