import os
import pathlib
from importlib import import_module
from inspect import isclass, isfunction, isroutine, getmembers

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree


class TreeBuilder:
    IGNORE_PREFIX = [".", "__", "test"]
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

                    bag = getmembers(import_module(path.__str__().replace("\\", ".")[:-3]))
                    for n, v in bag:
                        if os.path.splitext(n)[-1]:
                            self.add_leaf(path.name, branch)
                        else:
                            if isroutine(v):
                                if isfunction(v):
                                    self.add_leaf(n, branch, icon="📄 ")
                                    # self.add_leaf(n, branch, icon='⬛ ')
                            elif isclass(v):
                                self.add_leaf(n, branch, icon="📘 ")
                                # self.add_leaf(n, branch, icon='◼ ')
                            else:
                                pass

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
        for ig in self.IGNORE_PREFIX:
            if name.startswith(ig):
                return False
        return True

    @classmethod
    def print_tree(cls, path):
        if cls._tree is None:
            cls._tree = cls(path)
        print(cls._tree.tree)


if __name__ == "__main__":
    TreeBuilder.print_tree("./zae_engine")
