"""Execute a read-only research notebook without requiring a Jupyter kernel or GUI."""

# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Image, display
from IPython.utils.capture import capture_output

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    """Preserve tables and plots as notebook outputs, failing on any code-cell error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    shell = InteractiveShell.instance()
    if args.run_directory:
        shell.user_ns["RUN_DIR_OVERRIDE"] = args.run_directory.resolve()

    def capture_figures(*_args: object, **_kwargs: object) -> None:
        for number in plt.get_fignums():
            figure = plt.figure(number)
            data = BytesIO()
            figure.savefig(data, format="png", dpi=120)
            display(Image(data=data.getvalue()))
            plt.close(figure)

    plt.show = capture_figures
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f"Notebook cell {count} failed") from error
        cell.execution_count = count
        cell.outputs = []
        for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if value:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
        for item in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
    nbformat.validate(notebook)
    output = args.output or args.notebook
    nbformat.write(notebook, output)
    print(f"Executed and saved {count} code cells: {output}")


if __name__ == "__main__":
    main()
