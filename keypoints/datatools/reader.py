import json
import numpy as np
from typing import Any, Dict, List, Tuple


def read_json(annot_path: str) -> Dict[str, Any]:
    data = {}
    with open(annot_path, 'r') as f:
        data = json.load(f)
    return data


def decode_annot(annot_path, num_keypoints) -> Dict[str, List[Tuple[float, float]]]:
    res: Dict[int, List[Tuple[float, float]]] = {}
    mask = []
    with open(annot_path) as f:
        d = json.loads(f.read())
        for label, kp in d.items():
            if kp['visibility'] == 2.0:
                res[int(label)] = (kp['x'], kp['y'])

    for idx in range(num_keypoints):
        if idx not in res:
            res[idx] = None
    return res, mask


def read_annot(annot_path: str) -> Dict[str, List[Tuple[float, float]]]:
    annot = read_json(annot_path)
    res = decode_annot(annot, 0)
    return res
