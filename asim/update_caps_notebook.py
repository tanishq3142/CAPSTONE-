import json
import os
import re

NOTEBOOK_PATH = 'caps.ipynb'

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = 0
    
    # 1. Update K=9 to K=6 (Global Replacement with Regex)
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            new_source = []
            for line_idx, line in enumerate(cell['source']):
                # Regex to match k=9 with potential spaces
                if re.search(r"k\s*=\s*9", line):
                    print(f"Found K=9 in cell {cell_idx}, line {line_idx}: {line.strip()}")
                    new_line = re.sub(r"k\s*=\s*9", "k=6", line)
                    new_source.append(new_line)
                    changes_made += 1
                else:
                    new_source.append(line)
            cell['source'] = new_source

    # 2. Update Confidence Logic
    # Search for a unique part of the string
    target_substring = "freqs.sort(descending=True).values[:top_n].sum().item()"
    
    new_logic = """    # New Confidence Logic (Weighted)
    sorted_freqs = freqs.sort(descending=True).values
    score = 0.0
    
    # 1st Color (> 25%)
    if len(sorted_freqs) > 0:
        score += 0.4 if sorted_freqs[0] > 0.25 else 0.25
    else:
        score += 0.25
        
    # 2nd Color (> 20%)
    if len(sorted_freqs) > 1:
        score += 0.3 if sorted_freqs[1] > 0.20 else 0.15
    else:
        score += 0.15
        
    # 3rd Color (> 15%)
    if len(sorted_freqs) > 2:
        score += 0.2 if sorted_freqs[2] > 0.15 else 0.10
    else:
        score += 0.10
        
    # 4th Color (> 10%)
    if len(sorted_freqs) > 3:
        score += 0.1 if sorted_freqs[3] > 0.10 else 0.05
    else:
        score += 0.05
        
    return score
"""

    confidence_updated = False
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            for line_idx, line in enumerate(cell['source']):
                if target_substring in line:
                    print(f"Found confidence logic in cell {cell_idx}, line {line_idx}: {line.strip()}")
                    cell['source'][line_idx] = new_logic
                    confidence_updated = True
                    changes_made += 1
                    break
            if confidence_updated:
                break

    if changes_made > 0:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f) # removed indent to match original compact if needed, but pretty print is safer for future
        print(f"Successfully updated {NOTEBOOK_PATH} with {changes_made} changes.")
    else:
        print("No changes were made.")

if __name__ == "__main__":
    update_notebook()
