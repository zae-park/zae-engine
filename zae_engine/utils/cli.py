import shutil

import click

from .tree_builder import TreeBuilder


@click.command()
@click.argument("command", "path")
def cli_run(command: str, path: str):
    if command == "tree":
        TreeBuilder.print_tree("./zae_engine")
    if command == "example":
        shutil.copy("./example_script.py", path)
    else:
        print(f'Invalid command "{feature}".')
