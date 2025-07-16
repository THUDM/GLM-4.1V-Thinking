import re
from typing import Dict, List, Optional, Union, Any


def is_properly_closed(text):
    quote_count = 0
    bracket_stack = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            if i > 0 and text[i-1] == '\\':
                i += 1
                continue
            quote_count += 1
        elif char == '[':
            bracket_stack.append('[')
        elif char == ']':
            if not bracket_stack or bracket_stack[-1] != '[':
                return False
            bracket_stack.pop()
        i += 1
    return quote_count % 2 == 0 and len(bracket_stack) == 0

def _extract_coordinates(params_str: str, param_name: str) -> Optional[List[int]]:
    """
    Helper function to extract coordinates from parameter string.
    
    Args:
        params_str: Parameter string to search in
        param_name: Name of the parameter containing coordinates
        
    Returns:
        List of [x, y] coordinates or None if extraction fails
    """
    coord_match = re.search(
        rf"{param_name}='\[(\d+),\s*(\d+)\]'", params_str)
    if coord_match:
        return [int(coord_match.group(1)), int(coord_match.group(2))]
    return None
    
def parse_action(action_text: str) -> Optional[Dict[str, Any]]:
    if not action_text:
        return None
        
    action_type_match = re.search(r'^(\w+)\(', action_text)
    if not action_type_match:
        return None
        
    action_type = action_type_match.group(1)
    params_str = re.search(r'\((.*)\)$', action_text)
    params_str = params_str.group(1) if params_str else ""
    
    result = {'action_type': action_type}
    
    element_info_match = re.search(r"element_info='(.*?)'", params_str)
    if element_info_match:
        result['element_info'] = element_info_match.group(1)
    
    if action_type in ['left_click', 'left_double_click', 'right_click', 'middle_click', 'hover']:
        coords = _extract_coordinates(params_str, "start_box")
        if coords:
            result['coordinates'] = coords
            
    elif action_type == 'left_drag':
        start_coords = _extract_coordinates(params_str, "start_box")
        end_coords = _extract_coordinates(params_str, "end_box")
        
        if start_coords and end_coords:
            result['start_coordinates'] = start_coords
            result['end_coordinates'] = end_coords
            
    elif action_type == 'scroll':
        coords = _extract_coordinates(params_str, "start_box")
        direction_match = re.search(r"direction='(\w+)'", params_str)
        step_match = re.search(r"step=(\d+)", params_str)
        
        if coords and direction_match:
            result['coordinates'] = coords
            result['direction'] = direction_match.group(1)
            if step_match:
                result['step'] = int(step_match.group(1))
                
    elif action_type == 'type':
        content_match = re.search(r"content='(.*?)'", params_str)
        if content_match:
            result['content'] = content_match.group(1)
            
    elif action_type == 'key':
        key_match = re.search(r"keys='(.*?)'", params_str)
        if key_match:
            result['keys'] = key_match.group(1)
            
    return result


def calculate_coordinate_similarity(coord1: List[int], coord2: List[int], tolerance: int = 20) -> float:
    if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2:
        return 0.0
    distance = ((coord1[0] - coord2[0]) ** 2 +
                (coord1[1] - coord2[1]) ** 2) ** 0.5
    if distance <= tolerance:
        return 1.0 - (distance / tolerance) * 0.2
    else:
        return max(0.0, 1.0 - (distance / 50))


def _edit_distance_similarity(s1: str, s2: str) -> float:
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n]
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    ed = edit_distance(s1, s2)
    return 1.0 - (ed / max_len)


def _jaccard_similarity(s1: str, s2: str) -> float:
    def get_ngrams(text, n=2):
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    ngrams1 = get_ngrams(s1.lower(), 2)
    ngrams2 = get_ngrams(s2.lower(), 2)
    if not ngrams1 and not ngrams2:
        return 1.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def calculate_text_similarity(s1: str, s2: str) -> float:
    s1_norm = s1.strip().lower()
    s2_norm = s2.strip().lower()

    len1, len2 = len(s1_norm), len(s2_norm)
    if len1 == 0 or len2 == 0:
        return 0.0
    length_ratio = min(len1, len2) / max(len1, len2)
    if length_ratio < 0.3:
        length_penalty = length_ratio
    else:
        length_penalty = 1.0
    
    def has_excessive_repetition(text, threshold=0.7):
        if len(text) < 10:
            return False
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        max_char_freq = max(char_counts.values())
        if max_char_freq / len(text) > threshold:
            return True
        for substr_len in [2, 3, 4]:
            if substr_len >= len(text):
                continue
            substr_counts = {}
            for i in range(len(text) - substr_len + 1):
                substr = text[i:i+substr_len]
                substr_counts[substr] = substr_counts.get(substr, 0) + 1
            if substr_counts:
                max_substr_freq = max(substr_counts.values())
                coverage = max_substr_freq * substr_len / len(text)
                if coverage > threshold:
                    return True
        return False
    repetition_penalty = 1.0
    if has_excessive_repetition(s1_norm) or has_excessive_repetition(s2_norm):
        repetition_penalty = 0.3
    
    edit_sim = _edit_distance_similarity(s1_norm, s2_norm)
    jaccard_sim = _jaccard_similarity(s1_norm, s2_norm)
    combined_sim = 0.6 * edit_sim + 0.4 * jaccard_sim
    final_score = combined_sim * length_penalty * repetition_penalty

    return min(1.0, max(0.0, final_score))

def extract_answer(response: str, question: Optional[str] = None) -> Optional[Dict]:
    match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', response)
    if not match:
        return None
    content = match.group(1).strip()
    if not is_properly_closed(content):
        return None
    return parse_action(content)


def judge(
    extracted_answer: Any,
    ground_truth: Any,
    question: Optional[str] = None,
    image_path=None,
) -> float:
    if not isinstance(extracted_answer, dict) or not isinstance(ground_truth, dict):
        return 0.0

    if extracted_answer.get('action_type') != ground_truth.get('action_type'):
        return 0.0
    action_type = extracted_answer['action_type']

    if action_type in ['left_click', 'left_double_click', 'right_click', 'middle_click', 'hover']:
        if 'coordinates' not in extracted_answer or 'coordinates' not in ground_truth:
            return 0.0
        else:
            return calculate_coordinate_similarity(
                extracted_answer['coordinates'], ground_truth['coordinates'])

    elif action_type == 'left_drag':
        if not all(k in extracted_answer and k in ground_truth 
                 for k in ['start_coordinates', 'end_coordinates']):
            return 0.0
        else:
            start_sim = calculate_coordinate_similarity(
                extracted_answer['start_coordinates'], ground_truth['start_coordinates'])
            end_sim = calculate_coordinate_similarity(
                extracted_answer['end_coordinates'], ground_truth['end_coordinates'])
            return (start_sim * end_sim) ** 0.5

    elif action_type == 'scroll':
        if not all(k in extracted_answer and k in ground_truth for k in ['coordinates', 'direction']):
            return 0.0
        else:
            coord_sim = calculate_coordinate_similarity(
                extracted_answer['coordinates'], ground_truth['coordinates'])
            direction_match = 1.0 if extracted_answer['direction'] == ground_truth['direction'] else 0.0
            if 'step' in ground_truth:
                extracted_step = extracted_answer.get('step', 5)
                step_diff = abs(extracted_step - ground_truth['step'])
                step_sim = max(0.0, 1.0 - step_diff / (ground_truth['step'] * 2))
            else:
                step_sim = 1.0
            return coord_sim * direction_match * step_sim

    elif action_type == 'type':
        if 'content' not in extracted_answer or 'content' not in ground_truth:
            return 0.0
        else:
            return calculate_text_similarity(extracted_answer['content'], ground_truth['content'])

    elif action_type in ['key', 'hotkey']:
        if 'keys' not in extracted_answer or 'keys' not in ground_truth:
            return 0.0
        return 1.0 if extracted_answer['keys'].lower() == ground_truth['keys'].lower() else 0.0

    elif action_type.upper() in ['WAIT', 'DONE', 'FAIL'] or action_type.lower() in ['wait', 'finished']:
        return 1.0

    return 0.0
