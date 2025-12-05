from typing import Dict, Set

# Mapping: mistaken char -> correct char(s) it can be
# If multiple possibilities exist, we treat them as "equivalent group"
# CONFUSABLE_GROUPS: Dict[str, Set[str]] = {
#     # Your observed errors
#     '8': {'8', 'B'},
#     'B': {'8', 'B'},
#     '9': {'9', 'g'},
#     'g': {'9', 'g'},
#     '0': {'0', 'O', 'o'},
#     'O': {'0', 'O', 'o'},
#     'o': {'0', 'O', 'o'},
#     '1': {'1', 'l', 'I'},
#     'l': {'1', 'l', 'I'},
#     'I': {'1', 'l', 'I'},
#     '5': {'5', 'S'},
#     'S': {'5', 'S'},
#     '2': {'2', 'Z'},
#     'Z': {'2', 'Z'},
#     '+': {'+', '*'},
#     '*': {'+', '4'},
#     '.': {',', '.',' '},
# }

CONFUSABLE_GROUPS: Dict[str, Set[str]] = {
    
    '9': {'g'},
    '1': {'l', 'I'},
    '5': {'S'},
    '+': {'*'},
    'B': {'8'},
    '-': {'/'},
    # '.': {',', '.',' '},
}


def normalize_text(text: str) -> str:
    """
    Replace every character with a single canonical representative of its confusion group.
    Characters not in any group stay unchanged.
    """
    result = []
    for char in text:
        replaced = False
        for canonical, group in CONFUSABLE_GROUPS.items():
            if char in group:
                result.append(canonical)
                replaced = True
                break
        if not replaced:
            result.append(char)
    return ''.join(result)


def ocr_tolerant_compare(actual_str: str, detected_str: str) -> bool:
    norm_actual = normalize_text(actual_str)
    norm_detected = normalize_text(detected_str)
    print(f'    detected: {norm_detected}')
    print(f'    Actual: {norm_actual}')
    return norm_detected in norm_actual


def find_ocr_errors(actual_str: str, detected_str: str) -> list:

    norm_actual = normalize_text(actual_str)
    norm_detected = normalize_text(detected_str)

    errors_list = []
    max_len = max(len(norm_actual), len(norm_detected))
    for i in range(max_len):
        a = norm_actual[i] if i < len(norm_actual) else None
        d = norm_detected[i] if i < len(norm_detected) else None
        if a != d:
            orig_a = actual_str[i] if i < len(actual_str) else None
            orig_d = detected_str[i] if i < len(detected_str) else None
            errors_list.append((i, orig_a, orig_d))

    return errors_list


def fully_covered(actual_str: str, detected_list: list[str]) -> bool:
    remaining = normalize_text(actual_str)
    for frag in detected_list:
        norm_frag = normalize_text(frag).strip()
        if norm_frag:
            pos = remaining.find(norm_frag)
            if pos == -1:
                return False
            remaining = remaining[:pos] + remaining[pos + len(norm_frag):]
    # Allow only whitespace / noise left
    return len(remaining.strip()) <= 10  # tweak threshold as needed


def strip_matched_fragments(actual_list: list[str], detected_list: list[str]) -> list[str]:
    remaining_list=[
        fragment for fragment in actual_list
        if not any(fragment in detected_text for detected_text in detected_list)
    ]
    return remaining_list
