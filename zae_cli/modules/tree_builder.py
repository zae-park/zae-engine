import os
import ast
import pathlib
from collections import defaultdict

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

import ast
from collections import defaultdict


class NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_stack = []
        self.methods = defaultdict(list)

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        if node.name.startswith("__") or node.name.endswith("__") or "wrap" in node.name:
            return
        if self.class_stack:
            class_name = self.class_stack[-1]
            self.methods[class_name].append(node.name)
        else:
            self.methods["standalone"].append(node.name)


def get_methods_from_abspath(abs_path):
    try:
        with open(abs_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=abs_path)
    except Exception as e:
        print(f"Error reading file {abs_path}: {e}")
        return {}

    visitor = NodeVisitor()
    visitor.visit(tree)
    return visitor.methods


class TreeBuilder:
    START_IGNORE = [".", "__", "test"]
    _tree = None

    def __init__(self, root: str):
        self.tree = None
        self.set_tree(root)
        self.walk_in(pathlib.Path(root), self.tree)

    def set_tree(self, root):
        self.tree = Tree(f":open_file_folder: [link file://{root}]{root}", guide_style="bold bright_blue")

    def walk_in(self, root, tree):
        paths = self.get_path(root)
        for path in paths:
            if self.is_valid(path):
                if path.is_dir():
                    branch = tree.add(
                        f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                        style="",
                        guide_style="",
                    )
                    self.walk_in(path, branch)
                elif path.name.endswith(".py"):
                    branch = tree.add(
                        f"[bold magenta]🐍 [link file://{path}]{escape(path.name)}", style="", guide_style=""
                    )
                    methods = get_methods_from_abspath(str(path.resolve()))
                    for cls_name, funcs in methods.items():
                        if cls_name == "standalone":
                            for func_name in funcs:
                                self.add_leaf(func_name, branch, icon="📄 ")
                        else:
                            self.add_leaf(cls_name, branch, icon="📘 ")
                            for func_name in funcs:
                                self.add_leaf(func_name, branch, icon="📄 ")

    def add_branch(self, path, branch):
        text_filename = Text(path.name, "green")
        text_filename.highlight_regex(r"\..*$", "bold red")
        text_filename.stylize(f"link file://{path}")
        file_size = path.stat().st_size
        text_filename.append(f" ({decimal(file_size)})", "blue")
        icon = "🐍 " if path.suffix == ".py" else "📄 "
        branch.add(Text(icon) + text_filename)

    def add_leaf(self, name, branch, icon=None):
        text_filename = Text(name, "green")
        text_filename.highlight_regex(r"\..*$", "bold red")
        if icon is not None:
            text_filename = Text(icon) + text_filename
        branch.add(text_filename)

    def get_path(self, directory):
        key = lambda path: (path.is_file(), path.name.lower())
        paths = sorted(pathlib.Path(directory).iterdir(), key=key)
        return paths

    def is_valid(self, path):
        name = path.name
        return not any(name.startswith(ig) for ig in self.START_IGNORE)

    @classmethod
    def print_tree(cls, path):
        if cls._tree is None:
            cls._tree = cls(path)
        print(cls._tree.tree)


if __name__ == "__main__":
    TreeBuilder.print_tree("../../zae_engine")
