# homework/generate_qa.py
import json
from pathlib import Path
import random
import fire

# Keep this small to reduce token length and avoid OOM downstream
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400

def _parse_info(info_path: str):
    with open(info_path, "r") as f:
        return json.load(f)

def extract_frame_info_from_image(image_path: str) -> tuple[int, int]:
    """
    Parse image filename like 00048_09_im.jpg -> (frame_id_int, view_index)
    frame_id_int is the decimal value of the hex frame id.
    """
    name = Path(image_path).name
    parts = name.split("_")
    if len(parts) >= 2:
        try:
            frame_id = int(parts[0], 16)
        except Exception:
            # fallback: try decimal
            try:
                frame_id = int(parts[0])
            except Exception:
                frame_id = 0
        try:
            view_index = int(parts[1])
        except Exception:
            view_index = 0
        return frame_id, view_index
    return 0, 0

def extract_kart_objects(info_path: str, view_index: int, min_box_size: int = 5) -> list:
    """
    Return list of kart dicts:
      {
        "instance_id": int(track_id),
        "kart_name": str (from info['karts'] when available),
        "center": (cx, cy),
        "is_ego": bool (True if track_id==0 or chosen by center fallback)
      }
    Uses the raw detection coordinates as provided in info.json (600x400).
    """
    info = _parse_info(info_path)
    dets = info.get("detections", [])
    if view_index >= len(dets):
        return []

    frame_dets = dets[view_index]
    karts = []
    for det in frame_dets:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1:
            continue
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        name = None
        if "karts" in info:
            try:
                # some datasets use track_id as index into info["karts"]
                if 0 <= int(track_id) < len(info["karts"]):
                    name = info["karts"][int(track_id)]
            except Exception:
                name = None
        if not name:
            name = f"kart_{int(track_id)}"
        karts.append({
            "instance_id": int(track_id),
            "kart_name": name,
            "center": (cx, cy),
        })

    if not karts:
        return []

    # Prefer explicit track_id == 0 as ego if present
    ego_candidates = [k for k in karts if k["instance_id"] == 0]
    if ego_candidates:
        for k in karts:
            k["is_ego"] = (k["instance_id"] == 0)
        return karts

    # otherwise choose the one nearest image center (fallback)
    img_cx = ORIGINAL_WIDTH / 2.0
    img_cy = ORIGINAL_HEIGHT / 2.0
    for k in karts:
        dx = k["center"][0] - img_cx
        dy = k["center"][1] - img_cy
        k["dist_center_sq"] = dx * dx + dy * dy
    ego = min(karts, key=lambda x: x["dist_center_sq"])
    for k in karts:
        k["is_ego"] = (k is ego)

    # remove helper field
    for k in karts:
        k.pop("dist_center_sq", None)

    return karts

def _shorten_name(name: str) -> str:
    """
    Make names shorter where possible (not required, just keeps tokens tiny).
    """
    return name.replace("the_", "").replace("-", "_")

def generate_qa_pairs(info_path: str, view_index: int, max_relations: int = 4, sample_seed: int = 17) -> list:
    """
    Minimal QA generation:
      - What kart is the ego car?
      - How many karts are there?
      - What track is this?
      - For up to `max_relations` other karts: "Where is <NAME>?" -> "<vertical> and <horizontal>"
    Short and compact strings to reduce token length.
    """
    info = _parse_info(info_path)
    karts = extract_kart_objects(info_path, view_index)
    if not karts:
        return []

    # find ego
    ego = next((k for k in karts if k.get("is_ego")), None)
    if ego is None:
        return []

    ego_x, ego_y = ego["center"]

    qa = []
    # 1) Ego kart (short question)
    qa.append({
        "question": "What kart is the ego car?",
        "answer": _shorten_name(str(ego["kart_name"]))
    })

    # 2) Count
    qa.append({
        "question": "How many karts?",
        "answer": str(len(karts))
    })

    # 3) Track
    track_name = info.get("track", "unknown")
    qa.append({
        "question": "Which track?",
        "answer": str(track_name)
    })

    # 4) Relations: pick up to max_relations non-ego karts (deterministic sample)
    others = [k for k in karts if k is not ego]
    if not others:
        return qa

    # deterministic sampling
    rng = random.Random(sample_seed + view_index)
    rng.shuffle(others)
    others = others[:max_relations]

    for k in others:
        x, y = k["center"]
        horiz = "left" if x < ego_x else "right"
        vert = "front" if y < ego_y else "back"
        # condensed question + combined short answer in grader-friendly form
        qa.append({
            "question": f"Where is {_shorten_name(k['kart_name'])}?",
            "answer": f"{vert} and {horiz}"
        })

    return qa

def check(info_file: str, view_index: int = 0, max_relations: int = 4):
    """
    Fire command to visualize and print QA pairs for a single info file / view.
    """
    info_path = Path(info_file)
    if not info_path.exists():
        print("Info file not found:", info_file)
        return

    qa = generate_qa_pairs(str(info_path), int(view_index), max_relations=max_relations)
    print(json.dumps(qa, indent=2))

def main():
    fire.Fire({"check": check, "generate": generate_qa_pairs})

if __name__ == "__main__":
    main()
