import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

TRAIN_DIR = Path("data/train")

random.seed(17)

def load_info(path):
    """Load info.json and fallback to detections if no objects field exists."""
    with open(path, "r") as f:
        info = json.load(f)

    if "objects" not in info:
        dets = info.get("detections", [])
        if not dets:
            info["objects"] = []
        else:
            objs = []
            for det in dets[0]:
                class_id, track_id, x1, y1, x2, y2 = det
                objs.append({
                    "type": f"class_{int(class_id)}",
                    "bbox": [x1, y1, x2, y2]
                })
            info["objects"] = objs

    return info


def get_spatial_relations(objs):
    """
    Determine which object is front/back/left/right of the ego car.
    Using x-center only (SuperTuxKart scenes are roughly aligned).
    """
    if not objs:
        return {"front": "nothing", "back": "nothing",
                "left": "nothing", "right": "nothing"}

    centers = []
    for o in objs:
        x1, y1, x2, y2 = o.get("bbox", [0, 0, 0, 0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy, o["type"]))

    front = min(centers, key=lambda c: c[1])[2]
    back = max(centers, key=lambda c: c[1])[2]
    left = min(centers, key=lambda c: c[0])[2]
    right = max(centers, key=lambda c: c[0])[2]

    return {"front": front, "back": back, "left": left, "right": right}


def random_variant(options):
    return random.choice(options)

def generate_questions(info, view_img_file):
    objs = info["objects"]
    num_objs = len(objs)
    spatial = get_spatial_relations(objs)

    qas = []

    count_questions = [
        "How many objects are visible?",
        "How many items are present in the scene?",
        "Count the objects in this view.",
        "What is the total number of detected objects?"
    ]
    qas.append({
        "image_file": view_img_file,
        "question": random_variant(count_questions),
        "answer": str(num_objs)
    })

    spatial_templates = {
        "front": [
            "What object is ahead of the kart?",
            "What is in front of the kart?",
            "Which object lies forward?",
        ],
        "back": [
            "What object is behind the kart?",
            "Which object lies backward?",
            "What is located to the rear of the kart?",
        ],
        "left": [
            "What object is left of the kart?",
            "Which object appears on the left side?",
        ],
        "right": [
            "What object is right of the kart?",
            "Which object appears on the right side?",
        ]
    }

    for direction, templates in spatial_templates.items():
        qas.append({
            "image_file": view_img_file,
            "question": random_variant(templates),
            "answer": spatial[direction]
        })

    type_counts = {}
    for o in objs:
        t = o["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, ct in type_counts.items():
        qas.append({
            "image_file": view_img_file,
            "question": f"How many {t} objects are present?",
            "answer": str(ct)
        })

    caption_templates = [
        "Describe the scene.",
        "What is happening in the image?",
        "What objects can you see?",
        "Give a short caption for this scene.",
        "Summarize what appears in this view."
    ]
    caption_answer = ", ".join([o["type"] for o in objs]) if objs else "an empty road"

    qas.append({
        "image_file": view_img_file,
        "question": random_variant(caption_templates),
        "answer": caption_answer
    })

    return qas

def process_scene(info_file):
    scene_id = info_file.stem.replace("_info", "")

    # all images for scene
    images = sorted(TRAIN_DIR.glob(f"{scene_id}_*_im.jpg"))
    if not images:
        print(f"No images for {scene_id}, skipping.")
        return

    info = load_info(info_file)

    all_qas = []
    for img in images:
        rel_path = str(img.relative_to(TRAIN_DIR.parent))
        qas = generate_questions(info, rel_path)
        all_qas.extend(qas)

    out_path = TRAIN_DIR / f"{scene_id}_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"Wrote {out_path} with {len(all_qas)} questions.")


def main():
    for info_file in TRAIN_DIR.glob("*_info.json"):
        process_scene(info_file)


if __name__ == "__main__":
    main()
