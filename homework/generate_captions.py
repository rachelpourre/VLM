# homework/generate_captions.py
import json
from pathlib import Path

from homework.generate_qa import extract_kart_objects
from homework.generate_qa import ORIGINAL_WIDTH, ORIGINAL_HEIGHT


def generate_caption(info_path: str, view_index: int) -> list:
    """
    Minimal captions compatible with QA pipeline.
    Keeps captions short to avoid OOM in CLIP training.
    """
    info_path = Path(info_path)
    info = json.loads(info_path.read_text())

    track = info.get("track", "unknown")

    # reuse consistent object extraction
    karts = extract_kart_objects(str(info_path), view_index)
    if not karts:
        return [f"Empty scene on track {track}."]

    # find ego
    ego = next((k for k in karts if k["is_ego"]), None)
    ego_name = ego["kart_name"] if ego else "ego"

    captions = []

    # 1) Track
    captions.append(f"Scene on track {track}.")

    # 2) Ego
    captions.append(f"The ego kart is {ego_name}.")

    # 3) Count
    captions.append(f"There are {len(karts)} karts.")

    # 4) Relative short relations
    ego_x, ego_y = ego["center"]

    for k in karts:
        if k is ego:
            continue
        x, y = k["center"]
        horiz = "left" if x < ego_x else "right"
        vert = "front" if y < ego_y else "back"
        captions.append(f"{k['kart_name']} is {vert} and {horiz} of the ego.")

    return captions
