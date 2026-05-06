"""parsimmon: shared internal helpers."""

import sys

import numpy as np


class objdict(dict):
    """dict subclass with attribute-style access.

    Nested plain dicts are converted to objdict on construction so that
    ``d.key.subkey`` works at any depth.
    """

    def __init__(self, *args, **kwargs):
        # build via plain dict first, then convert any nested plain dicts
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if type(v) is dict:
                super().__setitem__(k, objdict(v))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'objdict' has no attribute {key!r}") from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'objdict' has no attribute {key!r}") from None

    def __repr__(self):
        return f"objdict({dict.__repr__(self)})"

    def copy(self):
        return objdict(self)


def _terminal_menu(prompt, options):
    """Interactive menu with arrow-key navigation. Returns selected index."""
    if not sys.stdin.isatty():
        return 0

    try:
        import termios
        import tty
    except ImportError:
        return 0

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    selected = 0
    n = len(options)

    def render():
        for i, opt in enumerate(options):
            if i == selected:
                sys.stdout.write(f"\r  \033[1m> {opt}\033[0m\033[K\n")
            else:
                sys.stdout.write(f"\r    {opt}\033[K\n")
        sys.stdout.flush()

    print(prompt)
    render()

    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    selected = (selected - 1) % n
                elif seq == "[B":
                    selected = (selected + 1) % n
            sys.stdout.write(f"\033[{n}A")
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return selected


def _to_nested_objdict(d):
    # objdict.__init__ already recurses into nested plain dicts, so wrapping is enough
    return objdict(d)


def _deep_update(base, updates):
    for key, val in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def _iter_leaves(d, prefix=()):
    for key, val in d.items():
        path = prefix + (key,)
        if isinstance(val, dict):
            yield from _iter_leaves(val, path)
        else:
            yield path, val


def _set_nested(d, path, val):
    for key in path[:-1]:
        d = d.setdefault(key, objdict())
    d[path[-1]] = val


def _get_nested(d, dotted_key):
    for part in dotted_key.split("."):
        if not isinstance(d, dict):
            raise KeyError(f"Cannot navigate into non-dict at '{part}' in '{dotted_key}'")
        if part not in d:
            raise KeyError(f"Key '{part}' not found while resolving '{dotted_key}'")
        d = d[part]
    return d


def _fmt_scalar(v):
    if isinstance(v, np.integer):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    return repr(v)
