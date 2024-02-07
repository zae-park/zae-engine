import shutil

import click

from .tree_builder import TreeBuilder


# https://ujuc.github.io/2019/08/15/writing_poetry_script/


def zae_print():
    print("zae-park")


@click.command()
@click.argument("command")
def cli_run(command: str, path: str):
    if command == "tree":
        TreeBuilder.print_tree("./zae_engine")
    elif command == "example":
        shutil.copy("./example_script.py", path)
    if command == "zae":
        zae_print()
    else:
        print(f'Invalid command "{command}".')
