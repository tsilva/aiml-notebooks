import os
import json
import re
import sys

COLAB_PREFIX = f"https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/"

def update_colab_link(cell_source, notebook_rel_path):
    # Replace markdown and HTML badge Colab links with the correct one
    new_md = f"[Open In Colab]({COLAB_PREFIX}{notebook_rel_path})"
    new_html = (
        f'<a href="{COLAB_PREFIX}{notebook_rel_path}" target="_parent">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    md_pattern = re.compile(r"\[open\s+in\s+colab\]\([^)]+\)", re.IGNORECASE)
    html_pattern = re.compile(
        r'<a\s+href="https://colab\.research\.google\.com/github/[^"]+"\s+target="_parent">\s*<img\s+src="https://colab\.research\.google\.com/assets/colab-badge\.svg"\s+alt="open in colab"\s*/?></a>',
        re.IGNORECASE,
    )
    new_lines = []
    for line in cell_source:
        # Replace markdown badge
        line = md_pattern.sub(new_md, line)
        # Replace HTML badge
        line = html_pattern.sub(new_html, line)
        new_lines.append(line)
    return new_lines

def process_notebook(nb_path, repo_root):
    with open(nb_path, "r", encoding="utf-8") as f: nb = json.load(f)

    changed = False
    rel_path = os.path.relpath(nb_path, repo_root)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown": continue
        
        src = cell.get("source", [])
        src_text = "".join(src)
        # Match markdown or HTML badge, case-insensitive
        if (
            re.search(r"\[open\s+in\s+colab\]", src_text, re.IGNORECASE)
            or re.search(r'colab-badge\.svg', src_text, re.IGNORECASE)
        ):
            new_src = update_colab_link(src, rel_path)
            if new_src == src: continue
            cell["source"] = new_src
            changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f: json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated: {nb_path}")

def main():
    # Search for notebooks in the sibling 'notebooks' folder
    root_dir = os.getcwd()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".ipynb"): continue
            nb_path = os.path.join(dirpath, fname)
            process_notebook(nb_path, root_dir)

if __name__ == "__main__":
    main()
