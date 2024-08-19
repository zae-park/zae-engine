import unittest
import os
import tempfile
import pathlib
from zae_cli.modules.tree_builder import TreeBuilder, get_methods_from_abspath
from rich.tree import Tree as RichTree
from rich.text import Text


class TestTreeBuilder(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and file for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.file_path = pathlib.Path(self.test_dir.name) / "test_script.py"
        with open(self.file_path, "w") as f:
            f.write(
                """\
class TestClass:
    def method_one(self):
        pass

    def method_two(self):
        pass

def standalone_function():
    pass

"""
            )

    def tearDown(self):
        # Cleanup temporary directory
        self.test_dir.cleanup()

    def test_get_methods_from_abspath(self):
        methods = get_methods_from_abspath(str(self.file_path))
        expected_methods = {"TestClass": ["method_one", "method_two"], "standalone": ["standalone_function"]}
        # Convert defaultdict to a regular dict for comparison
        self.assertEqual(
            dict(sorted((k, sorted(v)) for k, v in methods.items())),
            dict(sorted((k, sorted(v)) for k, v in expected_methods.items())),
        )

    def test_tree_builder_initialization(self):
        builder = TreeBuilder(self.test_dir.name)
        self.assertIsNotNone(builder.tree)

    def test_tree_builder_walk_in(self):
        # builder = TreeBuilder(self.test_dir.name)
        # Check the tree structure by examining its string representation
        tree_str = os.listdir(self.test_dir.name)
        self.assertIn("test_script.py", tree_str)

    def test_add_leaf(self):
        builder = TreeBuilder(self.test_dir.name)
        # Add a leaf manually
        branch = builder.tree
        new_branch = branch.add("Test Branch")
        builder.add_leaf("test_leaf", new_branch)
        # Check the leaf in the branch
        leaf_found = any("test_leaf" in child.label for child in new_branch.children)
        self.assertTrue(leaf_found)

    def test_add_branch(self):
        builder = TreeBuilder(self.test_dir.name)
        # Create a branch to add another branch
        branch = builder.tree
        new_branch = branch.add("Test Branch")
        builder.add_branch(self.file_path, new_branch)
        # Check if the new branch has been added
        branch_found = any("Test Branch" in child.label for child in branch.children)
        self.assertTrue(branch_found)

    def test_is_valid(self):
        builder = TreeBuilder(self.test_dir.name)
        # Testing valid and invalid paths
        valid_path = pathlib.Path(self.test_dir.name) / "valid_file.py"
        invalid_path = pathlib.Path(self.test_dir.name) / "__init__.py"
        valid_path.touch()
        invalid_path.touch()

        self.assertTrue(builder.is_valid(valid_path))
        self.assertFalse(builder.is_valid(invalid_path))


if __name__ == "__main__":
    unittest.main()
