# -*- coding: utf-8 -*-


import re
from collections.abc import Sequence
from typing import Optional

_BEGIN_OF_BOX = "<|begin_of_box|>"
_END_OF_BOX = "<|end_of_box|>"


def find_boxed_content_with_boxed(text: str) -> list[str]:
    """
    Extract all top-level \boxed{...} contents from the input string,
    handling nested braces correctly.

    This function scans the string manually to ensure proper brace matching,
    allowing for nested expressions inside \boxed{}, but only extracting
    the outermost matched segments.

    Behavior:
    - For nested boxed expressions like \\boxed{\\boxed{\\boxed{42}}},
      it returns ['\\boxed{\\boxed{42}}'] — treating the entire nested structure as one match.
    - For multiple separate boxed expressions like
      \\boxed{42} and \\boxed{42}, it returns ['42', '42'].

    Returns:
        A list of strings extracted from each top-level \boxed{...} block.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i : i + 7] == "\\boxed{":
            i += 7  # Move past "\boxed{"
            content = ""
            brace_count = 1
            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1

                if brace_count > 0:
                    content += text[i]
                i += 1

            results.append(content)
        else:
            i += 1
    return results


# NOTE:
# This is the original implementation of boxed content extraction.
# DAPO's version, which extracts only the innermost or final \boxed{...}, was evaluated
# but ultimately **not adopted**, as it encourages unwanted model behavior and overfitting.
# (Conclusion aligns with Mango's suggestion.)
# This function is the currently used version.
def find_boxed_content(text: str, begin_token: str = _BEGIN_OF_BOX, end_token: str = _END_OF_BOX) -> list[str]:
    """
    Extract all content between <|begin_of_box|> and <|end_of_box|> tokens from the input string.

    This function uses a stack-based approach similar to bracket matching to properly handle nested box tokens.

    Behavior:
    - For multiple boxed expressions like <|begin_of_box|>42<|end_of_box|> and <|begin_of_box|>43<|end_of_box|>,
      it returns ['42', '43'].
    - For nested box tokens, it will extract each complete pair separately.

    Returns:
        A list of strings extracted from each <|begin_of_box|>...<|end_of_box|> block.
    """
    # 1. find all \boxed{...}
    boxed_matches = find_boxed_content_with_boxed(text)
    if len(boxed_matches) > 0:
        return boxed_matches

    # 2. find all <|begin_of_box|>...<|end_of_box|>
    results = []

    stack: list[int] = []  # Stack to track begin token positions
    num = 0  # Counter for nested levels
    i = 0

    while i < len(text):
        # Find the next begin token
        begin_pos = text.find(begin_token, i)
        # Find the next end token
        end_pos = text.find(end_token, i)

        # If no more tokens found, break
        if begin_pos == -1 and end_pos == -1:
            break

        # If only end token found or end token comes before begin token
        if begin_pos == -1 or (end_pos != -1 and end_pos < begin_pos):
            if stack and end_pos != -1:
                num -= 1
                # Only extract content when we're back to the outermost level (num == 0)
                if num == 0:
                    # Extract content between the most recent begin token and current end token
                    content_start = stack[-1] + len(begin_token)
                    content = text[content_start:end_pos].strip()
                    results.append(content)

                # Pop the matched begin position from stack
                stack.pop()

            i = end_pos + len(end_token)
        else:
            # Push begin position to stack
            stack.append(begin_pos)
            num += 1
            i = begin_pos + len(begin_token)

    return results


def detect_long_paragraph_mixing(text: str, min_chinese_chars: int = 50, min_english_words: int = 200) -> bool:
    """
    检测文本中是否同时存在较长的中文段落和英文段落
    :param text: 输入文本
    :param min_chinese_chars: 中文段落最小字符数阈值
    :param min_english_words: 英文段落最小单词数阈值
    :return: bool (True表示存在长段混杂)
    """
    # 1. 按段落分割文本
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # 2. 检测标志
    has_long_chinese = False
    has_long_english = False

    # 3. 段落分析
    for para in paragraphs:
        # 跳过空段落
        if len(para) == 0:
            continue

        # 统计中文字符数 (Unicode 4E00-9FFF)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", para)
        chinese_count = len(chinese_chars)

        # 统计英文单词 (至少2个字母的连续序列)
        english_words = re.findall(r"\b[a-zA-Z]{2,}\b", para)
        english_count = len(english_words)

        # 计算段落总字符数 (用于比例计算)
        total_chars = len(para)

        # 判断中文段落 (中文字符占80%以上且达到最小长度)
        if chinese_count >= min_chinese_chars and chinese_count / total_chars > 0.8:
            has_long_chinese = True

        # 判断英文段落 (英文单词占70%以上且达到最小长度)
        if english_count >= min_english_words and english_count / (len(para.split()) + 1e-5) > 0.7:
            has_long_english = True

        # 如果已检测到两种长段落，提前返回
        if has_long_chinese and has_long_english:
            return True

    # 4. 最终判定
    return has_long_chinese and has_long_english


def detect_repeat(text: str, min_chars: int = 50, min_repetition: int = 10, exclude_length: int = 3) -> bool:
    """
    Detect repetitive content in text without using a tokenizer.

    Args:
        text (str): The input text to check for repetitions
        min_chars (int): Minimum character sequence length to consider
        min_repetition (int): Minimum number of repetitions to flag as repetitive
        exclude_length (int): Maximum length of repeated characters to exclude

    Returns:
        bool: True if repetitive content is detected, False otherwise
    """
    if min_chars <= 0:
        err_msg = f"`min_chars` should be greater than 0, but got {min_chars}."
        raise ValueError(err_msg)

    if min_repetition <= 1:
        err_msg = f"`min_repetition` should be greater than 1, but got {min_repetition}."
        raise ValueError(err_msg)

    def conditional_replace(content: str, char: str, max_length: int = 80) -> str:
        pattern = rf"{re.escape(char)}{{{max_length},}}"
        return re.sub(pattern, "", content)

    # Clean up the text by removing formatting patterns
    text = re.sub(r"\|[-]+\|", "|", text)
    for ch in ["=", "|", "-", "~", "_", "#", "*", ".", "%", "－", "█", " ", "─"]:
        text = conditional_replace(text, ch, max_length=exclude_length)

    # Use character-based sliding window instead of tokens
    times: dict[int, int] = {}
    for i in range(min_chars, len(text) + 1):
        sub = text[i - min_chars : i]
        hash_val = hash(sub)
        if hash_val in times:
            times[hash_val] += 1
            if times[hash_val] >= min_repetition:
                return True
        else:
            times[hash_val] = 1

    return False


def protect_template(template: str, allowed: Optional[Sequence[str]] = ("question", "predict", "label")) -> str:
    """
    Escape all {placeholders} in the template except those explicitly allowed.
    This protects against .format() crashes due to unintended format tokens in doc examples.
    """

    # Handle empty template
    if len(template) == 0:
        return ""

    # Escape all curly braces first
    escaped = template.replace("{", "{{").replace("}", "}}")

    # Restore allowed placeholders
    if allowed is not None:
        for field in allowed:
            pattern = f"{{{{{field}}}}}"
            replacement = f"{{{field}}}"
            escaped = escaped.replace(pattern, replacement)

    return escaped
