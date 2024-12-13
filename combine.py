import os
from pathlib import Path


def combine_python_files(directory: str, output_file: str = "combined_code.md"):
    """
    Find all Python files in the given directory and its subdirectories,
    and combine their contents into a single Markdown file.

    Args:
        directory (str): Root directory to search for Python files
        output_file (str): Name of the output Markdown file
    """
    # Convert directory to Path object
    root_dir = Path(directory)

    # Find all Python files
    python_files = list(root_dir.rglob("*.py"))

    # Sort files for consistent output
    python_files.sort()

    # Create or overwrite the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("# Combined Python Code\n\n")

        # Process each Python file
        for file_path in python_files:
            # Get relative path from root directory
            try:
                relative_path = file_path.relative_to(root_dir)
            except ValueError:
                relative_path = file_path

            # Write file header
            outfile.write(f"## {relative_path}\n\n")
            outfile.write("```python\n")

            # Read and write file contents
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)

                    # Ensure there's a newline at the end
                    if not content.endswith("\n"):
                        outfile.write("\n")
            except Exception as e:
                outfile.write(f"# Error reading file: {str(e)}\n")

            outfile.write("```\n\n")


if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()

    print(f"Searching for Python files in: {current_dir}")
    combine_python_files(current_dir)
    print("Done! Check combined_code.md for the output.")
