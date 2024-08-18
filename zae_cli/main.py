import os
import shutil
import click
from .modules import TreeBuilder
from .modules import run_doctor

# Package structure will be changed in the future.
# zae_cli
#   | ---- main
#   | ---- tree - tree_builder.py
#   | ---- example - generator.py example_script.py

# https://ujuc.github.io/2019/08/15/writing_poetry_script/


@click.group()
def cli():
    """Main entry point for the CLI."""
    pass


@click.command()
def hello():
    """Prints information about the user."""
    print("My name is zae-park")


@click.command()
@click.option("--path", type=click.Path(exists=False), default=None, help="Path to save the example file.")
def example(path):
    """Generates an example file at the specified path or current directory."""
    if path:
        # Use the specified path
        destination_path = path
    else:
        # Default to the current directory
        destination_path = os.path.join(os.getcwd(), "zae_example.py")

    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_script.py"), destination_path)
    print(f"Generated snippet file at {destination_path}")


@click.command()
def tree():
    """Prints the package structure."""
    TreeBuilder.print_tree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../zae_engine"))


@click.command()
@click.option("--verbose", is_flag=True, help="Show detailed output.")
def doctor(verbose):
    """Prints the installation status and system information."""
    run_doctor(verbose=verbose)


# Add commands to the CLI group
cli.add_command(hello)
cli.add_command(example)
cli.add_command(tree)
cli.add_command(doctor)


if __name__ == "__main__":
    cli()

# python -m zae_cli.cli
# print("module run")
# cli_run(["snippet"])
