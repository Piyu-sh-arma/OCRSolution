from typing import Dict, Set
# Module for OCR text comparison and normalization utilities.
# This module provides functions to handle OCR text comparison with tolerance for
# commonly confused characters. It helps identify and normalize visually similar
# characters that OCR engines often misrecognize.
# The main use cases are:
# - Normalizing text by replacing confusable characters with canonical representatives
# - Comparing OCR-detected text against expected text with character confusion tolerance
# - Filtering matched text fragments from a list

CONFUSABLE_GROUPS: Dict[str, Set[str]] = {
    
    '9': {'g'}, # 'g' often misread as '9'
    '1': {'l', 'I'}, # 'l' (lowercase L) and 'I' (uppercase i) often misread as '1'
    '5': {'S'}, # 'S' often misread as '5'
    '+': {'*'}, # '*' often misread as '+'
    'B': {'8'}, # '8' often misread as 'B'
    '-': {'/'}, # '/' often misread as '-'
}


def normalize_text(text: str) -> str:
    """
    Normalize text by replacing confusable characters with their canonical forms.
    This function iterates through each character in the input text and replaces
    characters that belong to a confusable group with their canonical representation.
    Characters not found in any confusable group are left unchanged.
    Args:
        text (str): The input text to be normalized.
    Returns:
        str: The normalized text with confusable characters replaced by their
             canonical forms.
    Example:
        >>> normalize_text("l1O")
        "lll"  # (assuming l, 1, O are in the same confusable group)
    Note:
        This function relies on the CONFUSABLE_GROUPS dictionary which defines
        character groups and their canonical representations.
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


def strip_matched_fragments(actual_list: list[str], detected_list: list[str]) -> list[str]:
    """
    Remove fragments from actual_list that are found in detected_list.
    Compares each fragment in actual_list against the detected_list, checking for matches
    both with and without spaces. Returns only the fragments from actual_list that do not
    appear in any of the detected items.
    Args:
        actual_list (list[str]): The list of expected/actual text fragments to filter.
        detected_list (list[str]): The list of detected text fragments to match against.
    Returns:
        list[str]: A list of fragments from actual_list that were not found in detected_list.
                   Fragments that match detected text (with or without spaces) are excluded.
    Example:
        >>> actual = ["Hello World", "Python Code", "Test"]
        >>> detected = ["HelloWorld", "Test"]
        >>> strip_matched_fragments(actual, detected)
        ['Python Code']
    """

    detected_list_without_spaces = [text.replace(" ", "") for text in detected_list]
    
    remaining_list = [
        fragment for fragment in actual_list
        if not any(
            fragment in detected_text or fragment in detected_text_no_space
            for detected_text, detected_text_no_space in zip(detected_list, detected_list_without_spaces)
        )
    ]
    return remaining_list
