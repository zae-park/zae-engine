import shutil

import click

from .tree_builder import TreeBuilder


@click.command()
@click.argument("feature", "path")
def cli_run(feature: str):
    if feature == "tree":
        TreeBuilder.print_tree("./")
    if feature in ["example", "sample", "snippet", "ex"]:
        shutil.copy("./example_script.py", path)
    else:
        print(f'Invalid command "{feature}".')
