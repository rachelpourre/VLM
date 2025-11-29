from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    info_path = Path(info_path)
    with open(info_path, "r") as f:
        info = json.load(f)

    kart_name = info.get("ego_kart", "the ego kart")
    track_name = info.get("track_name", "an unknown track")

    objects = info.get("objects", [])

    num_karts = len(objects)

    def get_center(obj):
        bbox = obj.get("bbox") or obj.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        return ( (x1 + x2) / 2, (y1 + y2) / 2 )

    centers = [(get_center(obj), obj["type"]) for obj in objects]

    if centers:
        front = min(centers, key=lambda t: t[0][1])[1]
        back = max(centers, key=lambda t: t[0][1])[1]
        left = min(centers, key=lambda t: t[0][0])[1]
        right = max(centers, key=lambda t: t[0][0])[1]
    else:
        front = back = left = right = "nothing"

    captions = []

    captions.append(f"{kart_name} is the ego car in this scene.")

    captions.append(f"There are {num_karts} karts in the scenario.")

    captions.append(f"The race takes place on {track_name}.")

    captions.append(f"In front of the ego car, you can see {front}.")
    captions.append(f"Behind the ego car, there is {back}.")
    captions.append(f"To the left side, there is {left}, and to the right side there is {right}.")

    readable_objects = ", ".join(o["type"] for o in objects) if objects else "nothing else"
    captions.append(f"Overall, the view contains the ego car along with {readable_objects}.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()