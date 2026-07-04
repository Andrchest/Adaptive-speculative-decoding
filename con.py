import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def combine_python_files(root_dir, output_file="all_code.txt", encoding="utf-8"):
    root = Path(root_dir)
    logger.info("Combining Python files from %s into %s", root, output_file)

    with open(output_file, "w", encoding=encoding) as out:
        for py_file in sorted(root.rglob("*.py")):
            try:
                relative_path = py_file.relative_to(root)
                logger.debug("Adding %s", relative_path)

                out.write(f"# {relative_path}\n")

                with open(py_file, encoding=encoding, errors="ignore") as f:
                    out.write(f.read())

                out.write("\n\n")

            except Exception as e:
                logger.warning("Skipping %s: %s", py_file, e)
                print(f"Skipping {py_file}: {e}")

    print(f"Combined code written to: {output_file}")
    logger.info("Combined code written to: %s", output_file)


if __name__ == "__main__":
    # Change this to your source directory
    source_directory = "./src/"

    combine_python_files(source_directory, "all_code.txt")
