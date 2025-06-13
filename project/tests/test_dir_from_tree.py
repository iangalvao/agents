import pytest  # type: ignore #VSCODE broken paths
from tools.dir_from_tree import parse_tree  # type: ignore #VSCODE broken paths


@pytest.fixture
def sample_tree():
    return """\
root/
    dir1/
    dir2/
        file1.txt
        file2.txt
    file3.txt
    # This is a comment
    dir3/
        subdir1/
            file4.txt
        subdir2/
            file5.txt
    # Another comment
    dir4/
        file6.txt
        file7.txt
    dir5/
        # This is an empty directory
        """


@pytest.fixture
def expected_structure():
    return {
        "root": {
            "dir1": {},
            "dir2": {"file1.txt": None, "file2.txt": None},
            "file3.txt": None,
            "dir3": {"subdir1": {"file4.txt": None}, "subdir2": {"file5.txt": None}},
            "dir4": {"file6.txt": None, "file7.txt": None},
            "dir5": {},
        }
    }


def test_parse_tree(sample_tree, expected_structure):
    tree_dict = parse_tree(sample_tree)
    assert (
        tree_dict == expected_structure
    ), f"Expected {expected_structure}, but got {tree_dict}"
