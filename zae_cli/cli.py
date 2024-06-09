import os
import shutil

import click

from .tree_builder import TreeBuilder


# Package structure will be changed in the future.
# zae_cli
#   | ---- main
#   | ---- tree - tree_builder.py
#   | ---- example - generator.py example_script.py

# https://ujuc.github.io/2019/08/15/writing_poetry_script/


def zae_print():
    print("My name is zae-park")


@click.command()
@click.argument("command", nargs=1)
@click.option(
    "path",
    "-p",
    type=click.Path(exists=False),
    help="The destination path where example file will be created when the `example` command.",
)
def cli_run(command: str, path: str = ""):
    if command == "tree":
        TreeBuilder.print_tree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../zae_engine"))

    elif command == "example":
        if not path:
            path = os.path.join(os.getcwd(), "zae_example.py")
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_script.py"), path)
        print(f"Generate snippet file. {path} python")
    elif command == "hello":
        zae_print()
    else:
        print(f'Invalid command "{command}".')


# if __name__ == "__main__":
#     os.system("zae tree")
# python -m zae_cli.cli
# print("module run")
# cli_run(["snippet"])
