import nbformat
from nbconvert import HTMLExporter

def convert_notebook_to_html(notebook_path, output_path):
    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)

    # Write the HTML output to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)


convert_notebook_to_html("repo/notebook_2.ipynb", "repo/Day_9.html")
