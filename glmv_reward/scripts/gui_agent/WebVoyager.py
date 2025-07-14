# reward_hook/math_verifier.py
import re
from typing import Any, Optional


def extract_answer(response: str, question: Optional[str] = None) -> Optional[str]:
    return re.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', response, re.DOTALL)[0]

def lcs(x, y):
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def judge(
    extracted_answer: Any,
    ground_truth: Any,
    question: Optional[str] = None,
    image_path = None,
) -> float:
    
    def extract_information(text):
        patterns = {
            "click": r"Click \[?(\d+)\]?",
            "type": r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
            "key":r"Key[; ]+\[?(.[^\]]*)\]?",
            "scroll": r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?",
            "wait": r"^Wait",
            "goback": r"^GoBack",
            "google": r"^Google",
            "bing": r"^Bing",
            "answer": r"ANSWER[; ]+<content>(.*?)</content>"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if key in ["click", "wait", "goback", "google","bing"]:
                    # no content
                    return key, match.groups()
                else:
                    return key, {"number": match.group(1), "content": match.group(2)} if key in ["type", "scroll"] else {"content": match.group(1)}
        return None, None


    
    action, info = extract_information(extracted_answer)
    gt_action, gt_info = extract_information(ground_truth) 
    if action != gt_action:
        return 0
    if action == "click":
        assert isinstance(info, tuple) and isinstance(gt_info, tuple)
        click_ele_number = int(info[0])
        gt_click_ele_number = int(gt_info[0])
        if click_ele_number == gt_click_ele_number:
            return 1.0
        else:
            return 0.0
    elif action=="type":
        assert isinstance(info, dict) and isinstance(gt_info, dict)
        type_ele_number = int(info['number'])
        gt_type_ele_number = int(gt_info['number'])
        input_text= info['content']
        gt_input_text= gt_info['content']
        if type_ele_number == gt_type_ele_number:
            return lcs(input_text, gt_input_text) / max(len(input_text), len(gt_input_text))
        else:
            return 0.0
    elif action=="key":
        assert isinstance(info, dict) and isinstance(gt_info, dict)
        key_name = info['content'].lower()
        gt_key_name = gt_info['content'].lower()
        if key_name == gt_key_name:
            return 1.0
        else:
            return 0.0
    elif action=="scroll":
        assert isinstance(info, dict) and isinstance(gt_info, dict)
        scroll_ele_number = info['number']
        scroll_content = info['content']
        gt_scroll_ele_number = gt_info['number']
        gt_scroll_content = gt_info['content']
        if scroll_ele_number == gt_scroll_ele_number:
            if scroll_content==gt_scroll_content:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0
    elif action=="wait":
        return 1.0
    elif action=="goback":
        return 1.0
    elif action=="google":
        return 1.0
    elif action=="bing":
        return 1.0
    elif action=="answer":
        assert isinstance(info, dict) and isinstance(gt_info, dict)
        answer_content = info['content']
        gt_answer_content = gt_info['content']
        return lcs(answer_content, gt_answer_content) / max(len(answer_content), len(gt_answer_content))
    else:
        return 0.0
