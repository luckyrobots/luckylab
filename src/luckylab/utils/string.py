"""String utilities for matching names and values.

Matches mjlab/third_party/isaaclab/isaaclab/utils/string.py.
"""

import re
from collections.abc import Sequence
from typing import Any


def resolve_matching_names_values(
    data: dict[str, Any], list_of_strings: Sequence[str], preserve_order: bool = False
) -> tuple[list[int], list[str], list[Any]]:
    """Match a list of regular expressions in a dictionary against a list of strings.

    Returns the matched indices, names, and values.

    Args:
        data: A dictionary of regular expressions and values to match.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of query keys.

    Returns:
        A tuple of lists containing the matched indices, names, and values.

    Raises:
        TypeError: When the input argument `data` is not a dictionary.
        ValueError: When multiple matches are found for a string.
        ValueError: When not all regular expressions are matched.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")

    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []

    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(data))]

    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if re.fullmatch(re_key, potential_match_string):
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                keys_match_found[key_index].append(potential_match_string)

    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1

        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]

        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder

    if not all(keys_match_found):
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found, strict=False):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        raise ValueError(
            f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
        )

    return index_list, names_list, values_list
