import os
import json
import re
import nbformat

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
    # Fix widget metadata first using nbformat
    fix_widget_metadata(nb_path)

    # Now process for Colab links
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
        # Use nbformat to write back if changes were made to links
        # Note: nbformat might reformat the JSON slightly
        nb_content = nbformat.reads(json.dumps(nb), as_version=nbformat.NO_CONVERT)
        nbformat.write(nb_content, nb_path)
        print(f"Updated Colab links: {nb_path}")
    # else: # No need for an else block, fix_widget_metadata already saved if it made changes
    #     print(f"No Colab link changes needed for: {nb_path}")


# Added function
def fix_widget_metadata(notebook_path):
    try:
        nb = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
        widgets = nb.metadata.get("widgets", None) # Use None to check existence

        # Only proceed if widgets metadata exists
        if widgets is not None:
            changed = False
            # Add empty widget state if missing or if state key is missing
            if "application/vnd.jupyter.widget-state+json" not in widgets:
                widgets["application/vnd.jupyter.widget-state+json"] = {
                    "state": {}
                }
                changed = True
            elif "state" not in widgets["application/vnd.jupyter.widget-state+json"]:
                widgets["application/vnd.jupyter.widget-state+json"]["state"] = {}
                changed = True

            if changed:
                nb.metadata["widgets"] = widgets
                nbformat.write(nb, notebook_path)
                print(f"✅ Fixed widget metadata: {notebook_path}")
            # else: # Optional: print if no widget fix was needed
            #     print(f"Widget metadata OK: {notebook_path}")
        # else: # Optional: print if no widget metadata section exists
        #     print(f"No widget metadata found in: {notebook_path}")

    except Exception as e:
        print(f"Error processing widget metadata for {notebook_path}: {e}")


def main():
    # Search for notebooks in the current directory and subdirectories
    root_dir = os.getcwd()
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip .ipynb_checkpoints directories
        if ".ipynb_checkpoints" in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith(".ipynb"): continue
            nb_path = os.path.join(dirpath, fname)
            try:
                process_notebook(nb_path, root_dir)
            except Exception as e:
                print(f"Failed to process {nb_path}: {e}") # Add error handling

if __name__ == "__main__":
    main()
