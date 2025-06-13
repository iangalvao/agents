import os
import math


"""
This script reads a tree structure from a text file and creates a directory structure
and empty files based on that structure. The tree format is expected to be similar to the output of the `tree` command.
The input file should contain lines formatted with indentation to represent directories and files.
Example input format:
├── dir1/
├── dir2/
│   ├── file1.txt
│   └── file2.txt
└── file3.txt

It should be called with two arguments:
1. The path to the text file containing the tree structure.
2. The path to the directory where the structure will be created.
# Usage:
python dir_from_tree.py path/to/tree.txt path/to/output_directory
"""


def remove_comments(tree_str):
    """
    Remove comments from the tree string.
    Comments are starts with '#' or empty lines, not granted that the # will be at line start.
    """
    lines = tree_str.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue  # Skip empty lines and comments
        # Remove trailing comments
        if "#" in stripped_line:
            line = stripped_line.split("#", 1)[0].rstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# Function to parse the tree format into a dictionary
def parse_tree(tree_str):
    """
    Parse the tree structure string into a nested dictionary.
    Each directory is represented as a dictionary, and files are represented as keys with None values.
    """
    tree_dict = {}
    indent_stack = [(-1, tree_dict)]  # Stack to keep track of current nesting level
    tree_str = remove_comments(tree_str)  # Clean the input string from comments

    for line in tree_str.splitlines():
        original_len = len(line)
        stripped_line = (
            line.lstrip("├── ")
            .lstrip("└── ")
            .lstrip("│   ")
            .replace("├── ", "")
            .replace("└── ", "")
            .replace("│   ", "")
        )
        if stripped_line:
            stripped_len = len(stripped_line)
            indent_level = math.floor((original_len - stripped_len) / 4)
            name = stripped_line.strip()

            # Pop stack to find the correct parent level
            while indent_stack and indent_stack[-1][0] >= indent_level:
                indent_stack.pop()

            # Get the current parent dictionary
            parent_level, parent_dict = indent_stack[-1]
            if name.endswith("/"):
                # It's a directory
                new_dict = {}
                parent_dict[name.strip("/")] = new_dict
                indent_stack.append((indent_level, new_dict))
            else:
                # It's a file
                parent_dict[name] = None

    return tree_dict


# Function to create directories and files from the dictionary
def create_from_dict(base_path, tree_dict):
    for name, content in tree_dict.items():
        path = os.path.join(base_path, name)
        if content is None:
            # It's a file
            open(path, "w").close()
        else:
            # It's a directory
            os.makedirs(path, exist_ok=True)
            create_from_dict(path, content)


# Main function
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a project structure from a tree representation."
    )
    parser.add_argument(
        "file", type=str, help="Path to the text file containing the tree structure."
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the directory where the structure will be created.",
    )
    args = parser.parse_args()

    # Read tree structure from file
    with open(args.file, "r") as file:
        tree_str = file.read()

    # Convert the tree string into a dictionary
    tree_dict = parse_tree(tree_str)

    # Create the project structure
    create_from_dict(args.output, tree_dict)


if __name__ == "__main__":
    main()
