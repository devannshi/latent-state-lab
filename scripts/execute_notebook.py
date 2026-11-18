"""Execute this repository's plain-Python tutorial cells without a Jupyter server.

This intentionally small validator supports the tutorial's explicit print/plot calls,
not arbitrary IPython magics or rich displays. Jupyter remains the normal interactive
way to use the notebook. Saved outputs accurately reflect these executed code cells.
"""

import base64
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "tutorial.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__", "__file__": str(path)}
    execution_count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        outputs = []

        def capture_figures(*args, **kwargs):
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(buffer.getvalue()).decode(),
                                         "text/plain": ["<Matplotlib figure>"]}})
                plt.close(figure)

        plt.show = capture_figures
        stream = io.StringIO()
        with redirect_stdout(stream):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
        if stream.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout",
                               "text": stream.getvalue().splitlines(keepends=True)})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    notebook["metadata"]["execution_note"] = (
        "All plain-Python code cells were executed by scripts/execute_notebook.py; "
        "no Jupyter kernel or IPython-specific syntax was used.")
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Executed {execution_count} tutorial cells: {path}")


if __name__ == "__main__":
    main()
