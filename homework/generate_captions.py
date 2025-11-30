# homework/generate_captions.py
import json
from pathlib import Path
from typing import List

from homework.generate_qa import extract_kart_objects
from homework.generate_qa import ORIGINAL_WIDTH, ORIGINAL_HEIGHT

# constants for scaled image size used throughout the repo
OUT_WIDTH = 150
OUT_HEIGHT = 100

def _join_captions(caps: List[str]) -> str:
    """
    Join a list of short caption sentences into a single caption string.
    Keeps sentences separated by a single space.
    """
    # strip and join, avoid double spaces
    return " ".join(s.strip() for s in caps if s is not None and str(s).strip())


def generate_caption_list(info_path: str, view_index: int) -> List[str]:
    """
    Generate a list of short caption strings for the given info file and view index.
    This preserves the older behavior (list of short sentences).
    """
    info_path = Path(info_path)
    info = json.loads(info_path.read_text())

    track = info.get("track", "unknown")

    # reuse consistent object extraction (returns karts with center, is_ego, kart_name)
    karts = extract_kart_objects(str(info_path), view_index)
    if not karts:
        return [f"Empty scene on track {track}."]

    # find ego (may be None)
    ego = next((k for k in karts if k.get("is_ego")), None)
    ego_name = ego["kart_name"] if ego else "ego"

    captions: List[str] = []

    # 1) Track
    captions.append(f"Scene on track {track}.")

    # 2) Ego
    captions.append(f"The ego kart is {ego_name}.")

    # 3) Count
    captions.append(f"There are {len(karts)} karts.")

    # 4) Relative short relations
    if ego:
        ego_x, ego_y = ego["center"]
    else:
        # fallback centers if no ego known
        ego_x, ego_y = OUT_WIDTH / 2.0, OUT_HEIGHT / 2.0

    for k in karts:
        if k is ego:
            continue
        x, y = k["center"]
        horiz = "left" if x < ego_x else "right"
        vert = "front" if y < ego_y else "back"
        captions.append(f"{k['kart_name']} is {vert} and {horiz} of the ego.")

    return captions


def generate_caption(info_path: str, view_index: int) -> str:
    """
    Produce one joined caption string for a given info file + view index.
    This is intended to be written into caption JSONs as the `caption` field
    (a single string) which the CLIP training pipeline expects.
    """
    caps_list = generate_caption_list(info_path, view_index)
    return _join_captions(caps_list)


# if you want to preserve a check-mode CLI locally, here's an optional helper:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--view_index", type=int, default=0)
    args = parser.parse_args()
    print(generate_caption(args.info_file, args.view_index))
