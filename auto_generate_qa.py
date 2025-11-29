import json
import random
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"

random.seed(17)

def load_info(path):
    """Load info.json. If no objects exist, convert detections."""
    with open(path, "r") as f:
        info = json.load(f)

    if "objects" not in info:
        dets = info.get("detections", [])
        objs = []
        if dets:
            for det in dets[0]:
                class_id, track_id, x1, y1, x2, y2 = det
                objs.append({
                    "type": f"class_{int(class_id)}",
                    "bbox": [x1, y1, x2, y2]
                })
        info["objects"] = objs

    return info

def get_center(obj):
    bbox = obj.get("bbox") or obj.get("box") or [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_spatial_relations(objs):
    if not objs:
        return {"front": "nothing", "back": "nothing",
                "left": "nothing", "right": "nothing"}

    centers = [(get_center(o), o["type"]) for o in objs]

    front = min(centers, key=lambda c: c[0][1])[1]
    back  = max(centers, key=lambda c: c[0][1])[1]
    left  = min(centers, key=lambda c: c[0][0])[1]
    right = max(centers, key=lambda c: c[0][0])[1]

    return {"front": front, "back": back, "left": left, "right": right}


def random_variant(options):
    return random.choice(options)

def generate_questions(info, rel_img_path):
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
        "image_file": rel_img_path,
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
            "image_file": rel_img_path,
            "question": random_variant(templates),
            "answer": spatial[direction]
        })

    type_counts = {}
    for o in objs:
        t = o["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, ct in type_counts.items():
        qas.append({
            "image_file": rel_img_path,
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
        "image_file": rel_img_path,
        "question": random_variant(caption_templates),
        "answer": caption_answer
    })

    return qas

def process_scene(info_file):
    scene_id = info_file.stem.replace("_info", "")

    images = sorted(TRAIN_DIR.glob(f"{scene_id}_*_im.jpg"))
    if not images:
        print(f"No images for {scene_id}, skipping.")
        return

    info = load_info(info_file)
    all_qas = []

    for img in images:
        rel_path = str(img.relative_to(DATA_DIR))  # gives "train/xxxx_im.jpg"

        qas = generate_questions(info, rel_path)
        all_qas.extend(qas)

    out_path = TRAIN_DIR / f"{scene_id}_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"Wrote {out_path} with {len(all_qas)} QA items.")


def main():
    print("Generating QA pairs for all scenes in data/train ...")
    for info_file in TRAIN_DIR.glob("*_info.json"):
        process_scene(info_file)
    print("Done.")


if __name__ == "__main__":
    main()