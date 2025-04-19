import os
import json
import re

REPO_PATH = "tsilva/aiml-notebooks"
GITHUB_BRANCH = "main"
COLAB_PREFIX = f"https://colab.research.google.com/github/{REPO_PATH}/blob/{GITHUB_BRANCH}/"

def update_colab_link(cell_source, notebook_rel_path):
    # Replace any existing colab link with the correct one
    def replacer(match):
        return f"[Open in Colab]({COLAB_PREFIX}{notebook_rel_path})"
    # Regex to match markdown colab links
    pattern = re.compile(r"\[Open in Colab\]\([^)]+\)")
    return [pattern.sub(replacer, line) for line in cell_source]

def process_notebook(nb_path, root_dir):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    rel_path = os.path.relpath(nb_path, root_dir)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", [])
            if any("Open in Colab" in line for line in src):
                new_src = update_colab_link(src, rel_path)
                if new_src != src:
                    cell["source"] = new_src
                    changed = True
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated: {nb_path}")

def main():
    root_dir = "/home/tsilva/repos/tsilva/aiml-notebooks"
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".ipynb"):
                nb_path = os.path.join(dirpath, fname)
                process_notebook(nb_path, root_dir)

if __name__ == "__main__":
    main()
