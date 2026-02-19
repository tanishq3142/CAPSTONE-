import json

def inspect():
    with open('caps.ipynb', 'r') as f:
        nb = json.load(f)
    
    with open('notebook_dump.txt', 'w') as f:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                f.write(f"--- CANDIDATE CELL {i} ---\n")
                # Source can be a string or list of strings
                source = cell['source']
                if isinstance(source, str):
                    f.write(source)
                else:
                    for line in source:
                        f.write(f"{repr(line)}\n")
                f.write("\n\n")

if __name__ == '__main__':
    inspect()
