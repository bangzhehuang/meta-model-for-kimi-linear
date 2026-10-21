"""Render a DSL graph's lowered recurrence as a readable equation.

Uses lower.normalize, which recognizes S_t = S_{t-1} Diag(d) + sum (S_{t-1} a) b^T + sum v k^T.
Feature names: q, k, v, alpha (= exp(g), channel decay), beta (scalar write gate).
'*' is channelwise multiplication; <a, b> is an inner product.
"""
import sys

from lower import ONE, ZERO, normalize


def expression(e):
    kind = e[0]
    if kind == 'number':
        value = e[1]
        return str(int(value)) if float(value).is_integer() else repr(value)
    if kind == 'input':
        return e[1]
    if kind == 'dot':
        return '<' + expression(e[1]) + ', ' + expression(e[2]) + '>'
    if kind == 'mul':
        number = [x for x in e[1:] if x[0] == 'number']
        factors = [x for x in e[1:] if x[0] != 'number']
        text = '*'.join(('(' + expression(x) + ')') if x[0] == 'add' else expression(x) for x in factors)
        if number:
            c = number[0][1]
            if c == -1: return '-' + text
            return expression(number[0]) + '*' + text
        return text
    if kind == 'add':
        text = expression(e[1])
        for x in e[2:]:
            term = expression(x)
            text += (' - ' + term[1:]) if term.startswith('-') else (' + ' + term)
        return text
    raise ValueError(e)


def factor(e):
    text = expression(e)
    return '(' + text + ')' if e[0] in ('add', 'mul') or text.startswith('-') else text


def equation(graph):
    m = normalize(graph)
    terms = []
    if m.diagonal == ONE: terms.append('S_{t-1}')
    elif m.diagonal != ZERO: terms.append('S_{t-1} Diag(' + expression(m.diagonal) + ')')
    for a, b in m.ranks:
        terms.append('(S_{t-1} ' + factor(a) + ') ' + factor(b) + '^T')
    for v, k in m.offsets:
        terms.append('[' + expression(v) + '] ' + factor(k) + '^T')
    return 'S_t = ' + ' + '.join(terms) + ';  y_t = S_t q_t / sqrt(d)'


if __name__ == '__main__':
    import json
    for line in open(sys.argv[1]):
        row = json.loads(line)
        print(row['id'][:12], equation(row['graph']))
