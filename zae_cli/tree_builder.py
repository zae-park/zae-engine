import os
import ast
import pathlib
from collections import defaultdict
from importlib import import_module
from inspect import isclass, isfunction, isroutine, getmembers

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree


def get_methods_from_abspath(abs_path):
    with open(abs_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=abs_path)

    defined_methods_with_classes = defaultdict(list)
    current_class = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            current_class = node.name
            defined_methods_with_classes[current_class] = []
        elif isinstance(node, ast.FunctionDef):
            # Check if the function is not a dunder
            if node.name.startswith("__") or node.name.endswith("__") or "wrap" in node.name:
                continue
            # Check if the function is in a class
            if current_class:
                defined_methods_with_classes[current_class].append(node.name)
            # Check if the function is standalone and not in a class
            else:
                defined_methods_with_classes["standalone"].append(node.name)

    return defined_methods_with_classes


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
                    # if current path is directory, add branch to tree and walk in to.
                    branch = tree.add(
                        f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                        style="",
                        guide_style="",
                    )
                    self.walk_in(path, branch)
                elif path.name.endswith(".py"):
                    # if current path is script, add branch to tree and find routine.
                    branch = tree.add(
                        f"[bold magenta]🐍 [link file://{path}]{escape(path.name)}", style="", guide_style=""
                    )

                    print(path)
                    # bag = getmembers(import_module(path.name[:-3]))
                    bag = get_methods_from_abspath(str(path.resolve()))
                    # bag = getmembers(import_module(path.__str__().replace("\\", ".")[:-3]))
                    for n, v in bag.items():
                        if os.path.splitext(n)[-1]:
                            self.add_leaf(path.name, branch)
                        else:
                            if n == "standalone":
                                for func_name in v:
                                    self.add_leaf(func_name, branch, icon="📄 ")
                            else:
                                # for cls_name in v:
                                self.add_leaf(n, branch, icon="📘 ")

                            # if isroutine(v):
                            #     if isfunction(v):
                            #         self.add_leaf(n, branch, icon="📘 ")
                            #         # self.add_leaf(n, branch, icon='⬛ ')
                            # elif isclass(v):
                            #     self.add_leaf(n, branch, icon="📘 ")
                            #     # self.add_leaf(n, branch, icon='◼ ')
                            # else:
                            #     pass

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
        for ig in self.START_IGNORE:
            if name.startswith(ig):
                return False
        return True

    @classmethod
    def print_tree(cls, path):
        if cls._tree is None:
            cls._tree = cls(path)
        print(cls._tree.tree)


if __name__ == "__main__":
    # res = get_methods_from_abspath("Z:\\dev-zae\\zae-engine\\zae_cli\\../zae_engine\\data_pipeline\\collate.py")
    TreeBuilder.print_tree("../zae_engine")
