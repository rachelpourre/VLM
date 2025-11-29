# generate_qa.py (FAST MINIMAL VERSION — OPTION A)

import json
from pathlib import Path
import fire


ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_karts(info_path, view_index, img_width=150, img_height=100):
    """Extract kart centers only. Minimal logic. No rendering."""
    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]

    karts = []
    ego = None

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        if class_id != 1:
            continue

        cx = ((x1 * scale_x) + (x2 * scale_x)) / 2
        cy = ((y1 * scale_y) + (y2 * scale_y)) / 2

        entry = {
            "id": track_id,
            "name": f"kart_{track_id}",
            "cx": cx,
            "cy": cy,
        }

        if track_id == 0:
            ego = entry

        karts.append(entry)

    if ego is None and karts:
        # fallback: choose the most central kart
        ex, ey = img_width / 2, img_height / 2
        ego = min(
            karts,
            key=lambda k: (k["cx"] - ex) ** 2 + (k["cy"] - ey) ** 2,
        )

    # mark ego
    for k in karts:
        k["is_ego"] = (k is ego)

    return karts


def extract_track(info_path):
    with open(info_path) as f:
        info = json.load(f)
    return info.get("map_name") or info.get("track") or "unknown"


def generate_qa(info_path, view_index):
    karts = extract_karts(info_path, view_index)
    track = extract_track(info_path)

    ego = next(k for k in karts if k["is_ego"])
    ex, ey = ego["cx"], ego["cy"]

    qa = []

    # 1 — ego
    qa.append({
        "question": "What kart is the ego car?",
        "answer": ego["name"],
    })

    # 2 — count
    qa.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
    })

    # 3 — track
    qa.append({
        "question": "What track is this?",
        "answer": track,
    })

    # relative counts
    left = right = front = behind = 0

    # individual kart relative QAs
    for k in karts:
        if k["is_ego"]:
            continue

        lr = "left" if k["cx"] < ex else "right"
        fb = "in front" if k["cy"] < ey else "behind"

        if lr == "left": left += 1
        else: right += 1

        if fb == "in front": front += 1
        else: behind += 1

        qa.append({
            "question": f"Is {k['name']} to the left or right of the ego car?",
            "answer": lr,
        })
        qa.append({
            "question": f"Is {k['name']} in front of or behind the ego car?",
            "answer": fb,
        })
        qa.append({
            "question": f"Where is {k['name']} relative to the ego car?",
            "answer": f"{lr} and {fb}",
        })

    # final count QAs
    qa.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left),
    })
    qa.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right),
    })
    qa.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front),
    })
    qa.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind),
    })

    return qa


def write(info_file, view_index, output_dir):
    info_file = Path(info_file)
    base = info_file.stem.replace("_info", "")
    out_path = Path(output_dir) / f"{base}_qa_pairs.json"

    qa = generate_qa(info_file, int(view_index))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qa, f, indent=2)

    print("Wrote:", out_path)


def main():
    fire.Fire({"write": write})


if __name__ == "__main__":
    main()
