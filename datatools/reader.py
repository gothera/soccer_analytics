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
        for shape in d['shapes']:
            label = shape['label']
            points = list(np.array(shape['points']))
            if label == 'center':
                pass
            elif label == 'right':
                pass
            elif label == 'left':
                pass
            else:
                res[int(label)] = (points[0][0], points[0][1])

    for idx in range(num_keypoints):
        if idx not in res:
            res[idx] = None
        if idx > 29 and idx not in res:
            mask.append(idx)
    return res, mask


def read_annot(annot_path: str) -> Dict[str, List[Tuple[float, float]]]:
    annot = read_json(annot_path)
    res = decode_annot(annot, 0)
    return res
