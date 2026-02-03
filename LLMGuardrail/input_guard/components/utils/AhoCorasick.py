import json
import csv
from collections import deque, defaultdict


class Node:
    def __init__(self):
        self.fail = None
        self.children = dict()
        self.word = None

def build_ac_automaton(keywords):
    """构建AC自动机"""
    root = Node()
    for word in keywords:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
        node.word = word
    
    queue = deque([root])
    while queue:
        current_node = queue.popleft()
        for char, child in current_node.children.items():
            if current_node == root:
                child.fail = root
            else:
                p = current_node.fail
                while p and char not in p.children:
                    p = p.fail
                child.fail = p.children[char] if p else root
            queue.append(child)
    return root

def find_occurrences(root, text):
    """执行模式匹配"""
    current = root
    occurrences = defaultdict(list)
    for i, char in enumerate(text):
        while current != root and char not in current.children:
            current = current.fail
        current = current.children[char] if char in current.children else root
        
        temp = current
        while temp != root:
            if temp.word:
                word_len = len(temp.word)
                start = i - word_len + 1
                if start >= 0 and text[start:i+1] == temp.word:
                    occurrences[temp.word].append((start, i+1))
            temp = temp.fail
    return occurrences

def load_keywords(shielding_lib, additional_keywords):
    """加载并标准化关键词"""
    keywords = []
    if isinstance(shielding_lib, str):
        try:
            if shielding_lib.endswith('.json'):
                with open(shielding_lib, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    keywords.extend(data if isinstance(data, list) else [])
            elif shielding_lib.endswith('.csv'):
                with open(shielding_lib, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    keywords.extend([col for row in reader for col in row if col])
            else:
                keywords.extend(shielding_lib.split(','))
        except FileNotFoundError:
            keywords.extend(shielding_lib.split(','))
    elif isinstance(shielding_lib, list):
        keywords.extend(shielding_lib)
    
    keywords.extend(additional_keywords)
    return list({word.strip() for word in keywords if isinstance(word, str) and word.strip()})

def replace_masked_text(text, occurrences, replace_str):
    """执行文本替换"""
    all_positions = set()
    for positions in occurrences.values():
        all_positions.update(positions)
    if not all_positions:
        return text
    
    if len(replace_str) == 1:
        text_list = list(text)
        replace_char = replace_str
        for start, end in sorted(all_positions, key=lambda x: -x[0]):
            for i in range(start, end):
                text_list[i] = replace_char
        return ''.join(text_list)
    else:
        sorted_positions = sorted(all_positions, key=lambda x: x[0])
        merged = []
        for current in sorted_positions:
            if not merged:
                merged.append(current)
            else:
                last = merged[-1]
                if current[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)
        
        text_list = list(text)
        offset = 0
        for start, end in merged:
            actual_start = start + offset
            actual_end = end + offset
            original_length = end - start
            new_length = len(replace_str)
            del text_list[actual_start:actual_end]
            text_list[actual_start:actual_start] = list(replace_str)
            offset += new_length - original_length
        return ''.join(text_list)

def ac_text_filter(text, shielding_lib, additional_keywords, replace_str):
    keywords = load_keywords(shielding_lib, additional_keywords)
    if not keywords:
        return text, {}
    root = build_ac_automaton(keywords)
    occurrences = find_occurrences(root, text)
    stats = {}
    for word, positions in occurrences.items():
        unique_pos = sorted(set(positions))
        stats[word] = {'count': len(unique_pos), 'positions': unique_pos}
    return replace_masked_text(text, occurrences, replace_str), stats

