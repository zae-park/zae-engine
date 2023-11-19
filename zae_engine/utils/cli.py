import click
from .tree_builder import TreeBuilder


@click.command()
@click.argument('feature')
def cli_run(feature: str):
    if feature == 'tree':
        TreeBuilder.print_tree('./')
    else:
        print(f'Invalid command "{feature}".')
