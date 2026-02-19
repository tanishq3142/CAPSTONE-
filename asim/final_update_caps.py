import json

NOTEBOOK_PATH = "caps.ipynb"

def update():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changes = 0
    conf_replaced = False
    
    # Target string for confidence logic (stripped for matching)
    target_conf_stripped = "return freqs.sort(descending=True).values[:top_n].sum().item()"

    new_conf_block = """    # New Confidence Logic (Weighted)
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

    for c_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
            
        source = cell["source"]
        new_source = []
        
        # Determine if source is list or string (nbformat 4 usually list)
        if isinstance(source, str):
            source_lines = source.splitlines(keepends=True)
        else:
            source_lines = source
            
        for l_idx, line in enumerate(source_lines):
            # 1. Update k=9 -> k=6
            if "k=9" in line:
                print(f"[K-UPDATE] Cell {c_idx} Line {l_idx}: {line.strip()}")
                line = line.replace("k=9", "k=6")
                changes += 1
                
            # 2. Update Confidence Logic
            if target_conf_stripped in line.strip():
                print(f"[CONF-UPDATE] Cell {c_idx} Line {l_idx}: {line.strip()}")
                # Preserve indentation?
                # The line usually has 4 spaces indent.
                # My new block has some indentation, but maybe I should just use the raw block
                # and let it rely on the fact that it's Python code.
                # The prompt block has 4 spaces indent for the first line.
                line = new_conf_block
                conf_replaced = True
                changes += 1
            
            new_source.append(line)
        
        cell["source"] = new_source

    if changes > 0:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f) # Write compact to match original style if preferred, or indent=1
        print(f"Saved {changes} changes to {NOTEBOOK_PATH}")
    else:
        print("No changes found!")

if __name__ == "__main__":
    update()
