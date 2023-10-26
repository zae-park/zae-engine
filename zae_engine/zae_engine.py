import os
import click




@click.command()
@click.argument('sub_command', nargs=1)
@click.argument('args', nargs=-1)
def fake(sub_command, args):
    """
    These are common Fakepack commands used in various situations:

    Authorization
        init        Resistrate author's information (or token).
    
    Manage dependency
        add         Resistrate author's information (or token).

    - init
    - add
    - publish
    """
    assert sub_command in ['init', 'add', 'publish'], f'The command {sub_command} does not exist.'
    
    print(f'{name} - {url}')


class Tokenizer:
    def __init__(self) -> None:
        pass

    def p(a):
        print(a)


class Manager: 
    def __init__(self) -> None:
        pass

    def m(a):
        print(a)




@click.command()
@click.argument('src', nargs=-1)
@click.argument('dst', nargs=1)
def copy(src, dst):
    """Move file SRC to DST."""
    for fn in src:
        click.echo('move %s to folder %s' % (fn, dst))


@click.command()
def dummy():
    """Move file SRC to DST."""
    print('dummy')
