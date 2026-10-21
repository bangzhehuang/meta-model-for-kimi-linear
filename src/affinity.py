"""Conservative affinity filter, separate from the DSL shape/budget validator."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    operation: str
    inputs: tuple[str, ...]


class AffinityError(ValueError):
    pass


PORTS = {'Sprev': True, 'alpha': False, 'beta': False,
         'q': False, 'k': False, 'v': False}
BILINEAR = {'MatMul', 'ScalarScale', 'ChannelScale'}
VIEWS = {'Transpose', 'Column', 'Row'}
SUMS = {'SignedSum', 'OutputSum'}


def affine_dependencies(nodes: list[Node]) -> dict[str, bool]:
    """Check a topologically ordered DAG, returning each source's dependency.

    A true flag conservatively means affine dependence on this cell's Sprev.
    Signs and shapes do not change dependency and are checked separately.
    """
    dependencies = dict(PORTS)
    for node in nodes:
        if node.name in dependencies:
            raise AffinityError('Duplicate or shadowed source: ' + node.name)
        if any(source not in dependencies for source in node.inputs):
            raise AffinityError('Unknown, cyclic or forward source at ' + node.name)
        flags = [dependencies[source] for source in node.inputs]
        if node.operation in BILINEAR:
            if len(flags) != 2:
                raise AffinityError('Bilinear operation requires two operands')
            if sum(flags) > 1:
                raise AffinityError('Product has two state-dependent operands: ' + node.name)
        elif node.operation in VIEWS:
            if len(flags) != 1:
                raise AffinityError('View requires one operand')
        elif node.operation in SUMS:
            if not flags:
                raise AffinityError('Sum requires at least one operand')
        else:
            raise AffinityError('Operation is outside the affine memory DSL: ' + node.operation)
        dependencies[node.name] = any(flags)
    return dependencies


def kda_graph() -> list[Node]:
    return [Node('F', 'ChannelScale', ('Sprev', 'alpha')),
            Node('r', 'MatMul', ('F', 'k')),
            Node('e', 'SignedSum', ('v', 'r')),
            Node('w', 'ScalarScale', ('beta', 'e')),
            Node('wc', 'Column', ('w',)), Node('kr', 'Row', ('k',)),
            Node('U', 'MatMul', ('wc', 'kr')),
            Node('St', 'OutputSum', ('F', 'U'))]
