import re
import json
from typing import Optional, Any

def extract_answer_obj(s: str):
    if '<|begin_of_box|>' not in s or '<|end_of_box|>' not in s:
        return None
    try:
        res = s.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0].strip()

        # Processing leading zeros if any
        ptn = r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]"
        m = re.search(ptn, res)
        if m:
            old_str = m.group(0)
            v1 = int(m.group(1))
            v2 = int(m.group(2))
            v3 = int(m.group(3))
            v4 = int(m.group(4))
            new_str = f"[[{v1},{v2},{v3},{v4}]]"
            res = res.replace(old_str, new_str)
        try:
            return json.loads(res)
        except:
            return eval(res, {
                'true': True, 
                'false': False,
                'null': None
            }) 
    except:
        return None


def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def iou(box1, box2):
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])

    iw = max(ixmax - ixmin, 0)
    ih = max(iymax - iymin, 0)
    inter_area = iw * ih

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def check_box_valid(box):
    if not box:
        return False
    if not isinstance(box, list) or len(box) != 4:
        return False
    for x in box:
        if not isinstance(x, int):
            return False
        if x < 0 or x > 999:
            return False
    
    if box[0] > box[2] or box[1] > box[3]:
        return False
    return True


def extract_answer(response: str, question: Optional[str] = None) -> Optional[dict]:
    return extract_answer_obj(response)

def judge(
    extracted_answer: Any,  
    ground_truth: Any, 
    question: Optional[str] = None,
    image_path = None,
) -> float:
    assert isinstance(ground_truth, dict)

    if not isinstance(extracted_answer, dict):
        return 0.0

    if set(extracted_answer.keys()) != set(ground_truth.keys()):
        # maybe format error or different action type
        return 0.0
    
    key2scores = {}
    for key in ground_truth:
        if extracted_answer[key] == ground_truth[key]:
            key2scores[key] = 1.0
            continue
        
        if key == "text":
            t1 = ground_truth[key]
            t2 = extracted_answer[key]
            key2scores[key] = lcs(t1, t2) / max(len(t1), len(t2))
        elif key == 'box_2d':
            # box in [[xmin,ymin,xmax,ymax]] format
            box1 = ground_truth[key][0]
            assert check_box_valid(box1)
            box2 = extracted_answer[key][0]
            if not check_box_valid(box2):
                return 0.0
            
            key2scores[key] = iou(box1, box2)
        else: # It must be an enumeration type.
            key2scores[key] = 0.0
    
    reward = 1
    for sub_score in key2scores.values():
            reward *= sub_score

    return reward

if __name__ == "__main__":
    gt_override = {'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'override': True, 'box_2d': [[38, 115, 960, 141]]}
    ans_override2 = {'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'override': True, 'box_2d': [[38, 115, 961, 140]]}

    print(judge(
        question="How many sub-pages are listed under the Sub-pages section?", 
        extracted_answer=ans_override2,
        ground_truth=gt_override
    ))