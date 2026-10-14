"""Tiny recursive-descent expression compiler for the rule DSL.

Grammar (Pratt-style precedence, left-associative except assignment which
does not exist):

.. code-block:: text

    expr    := ternary
    ternary := logical_or ('?' logical_or ':' ternary)?
    or      := and ('||' and)*
    and     := comp ('&&' comp)*
    comp    := add (('>' | '<' | '>=' | '<=' | '==' | '!=') add)?
    add     := mul (('+' | '-') mul)*
    mul     := unary (('*' | '/' | '%') unary)*
    unary   := ('-' | '!') unary | primary
    primary := number | string | ident_or_call | '(' expr ')'
    ident_or_call := identifier ('.' identifier)*  ( '(' args? ')' )?

Values at runtime are floats, strings, booleans, or ``None`` (unavailable).
Any arithmetic on ``None`` short-circuits to ``None`` — that propagates as
a false predicate, which is exactly what we want when a variable isn't
defined yet (warmup, missing profile, no ATR history).
"""

from __future__ import annotations

import math
from typing import Any, Callable


class ExprError(ValueError):
    pass


# ── tokenizer ────────────────────────────────────────────────────────────

_KEYWORDS = {"true", "false", "null"}
# Reserved bare-word string literals — so users can write
#   {"eq": ["C.color", "green"]}
# without quoting. The interpreter binds C.color to "green"/"red"/"doji".
_STRING_KEYWORDS = {"green", "red", "doji", "long", "short", "flat"}


def _tokenize(src: str) -> list[tuple[str, Any, int]]:
    toks: list[tuple[str, Any, int]] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            try:
                num = float(src[i:j])
            except ValueError as exc:
                raise ExprError(f"bad number at {i}: {src[i:j]!r}") from exc
            toks.append(("NUM", num, i))
            i = j
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n and src[j] != quote:
                j += 1
            if j >= n:
                raise ExprError(f"unterminated string at {i}")
            toks.append(("STR", src[i + 1: j], i))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] in ("_",)):
                j += 1
            word = src[i:j]
            if word in _KEYWORDS:
                if word == "true":
                    toks.append(("BOOL", True, i))
                elif word == "false":
                    toks.append(("BOOL", False, i))
                else:
                    toks.append(("NULL", None, i))
            elif word in _STRING_KEYWORDS:
                toks.append(("STR", word, i))
            else:
                toks.append(("IDENT", word, i))
            i = j
            continue
        two = src[i:i + 2]
        if two in ("==", "!=", ">=", "<=", "&&", "||"):
            toks.append(("OP", two, i))
            i += 2
            continue
        if ch in "+-*/%().,><?:":
            toks.append(("OP" if ch not in "()," else ch, ch, i))
            i += 1
            continue
        raise ExprError(f"unexpected char {ch!r} at {i}")
    toks.append(("END", None, n))
    return toks


# ── parser ───────────────────────────────────────────────────────────────

_ALLOWED_CALLS = {
    "min", "max", "abs", "floor", "ceil", "round",
    "atr", "ema", "sma",
    "next_up_of", "next_down_of",
    "lp_net", "lp_buy", "lp_sell", "lp_count", "lp_max_size",
}


class _Parser:
    def __init__(self, tokens: list[tuple[str, Any, int]], catalog: dict[str, Any] | None):
        self.toks = tokens
        self.pos = 0
        self.catalog = catalog

    def peek(self) -> tuple[str, Any, int]:
        return self.toks[self.pos]

    def eat(self, kind: str, val: Any = None) -> tuple[str, Any, int]:
        tok = self.toks[self.pos]
        if tok[0] != kind or (val is not None and tok[1] != val):
            raise ExprError(f"expected {kind}{'/' + str(val) if val else ''} at {tok[2]}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Any:
        node = self.expr()
        end = self.eat("END")
        return node

    def expr(self) -> Any:
        return self.ternary()

    def ternary(self) -> Any:
        cond = self.logical_or()
        tok = self.peek()
        if tok[0] == "OP" and tok[1] == "?":
            self.pos += 1
            a = self.logical_or()
            self.eat("OP", ":")
            b = self.ternary()
            return ("?:", cond, a, b)
        return cond

    def logical_or(self) -> Any:
        node = self.logical_and()
        while self.peek() == ("OP", "||", self.peek()[2]) if False else (self.peek()[0] == "OP" and self.peek()[1] == "||"):
            self.pos += 1
            rhs = self.logical_and()
            node = ("||", node, rhs)
        return node

    def logical_and(self) -> Any:
        node = self.comparison()
        while self.peek()[0] == "OP" and self.peek()[1] == "&&":
            self.pos += 1
            rhs = self.comparison()
            node = ("&&", node, rhs)
        return node

    def comparison(self) -> Any:
        node = self.additive()
        tok = self.peek()
        if tok[0] == "OP" and tok[1] in (">", "<", ">=", "<=", "==", "!="):
            op = tok[1]
            self.pos += 1
            rhs = self.additive()
            node = (op, node, rhs)
        return node

    def additive(self) -> Any:
        node = self.multiplicative()
        while self.peek()[0] == "OP" and self.peek()[1] in ("+", "-"):
            op = self.peek()[1]
            self.pos += 1
            rhs = self.multiplicative()
            node = (op, node, rhs)
        return node

    def multiplicative(self) -> Any:
        node = self.unary()
        while self.peek()[0] == "OP" and self.peek()[1] in ("*", "/", "%"):
            op = self.peek()[1]
            self.pos += 1
            rhs = self.unary()
            node = (op, node, rhs)
        return node

    def unary(self) -> Any:
        tok = self.peek()
        if tok[0] == "OP" and tok[1] in ("-", "!"):
            self.pos += 1
            rhs = self.unary()
            return (f"u{tok[1]}", rhs)
        return self.primary()

    def primary(self) -> Any:
        tok = self.peek()
        if tok[0] == "NUM":
            self.pos += 1
            return ("num", tok[1])
        if tok[0] == "STR":
            self.pos += 1
            return ("str", tok[1])
        if tok[0] == "BOOL":
            self.pos += 1
            return ("bool", tok[1])
        if tok[0] == "NULL":
            self.pos += 1
            return ("null",)
        if tok[0] == "(":
            self.pos += 1
            node = self.expr()
            self.eat(")")
            return node
        if tok[0] == "IDENT":
            return self.ident_or_call()
        raise ExprError(f"unexpected token {tok}")

    def ident_or_call(self) -> Any:
        parts: list[str] = []
        first = self.eat("IDENT")
        parts.append(str(first[1]))
        while True:
            tok = self.peek()
            if tok[0] == "OP" and tok[1] == ".":
                self.pos += 1
                nxt = self.eat("IDENT")
                parts.append(str(nxt[1]))
                continue
            break
        # function call?
        if self.peek()[0] == "(":
            self.pos += 1
            args: list[Any] = []
            if self.peek()[0] != ")":
                args.append(self.expr())
                while self.peek()[0] == ",":
                    self.pos += 1
                    args.append(self.expr())
            self.eat(")")
            fname = ".".join(parts).lower()
            if fname not in _ALLOWED_CALLS:
                raise ExprError(f"unknown function {fname!r}")
            return ("call", fname, args)
        # identifier
        name = ".".join(parts)
        if self.catalog is not None:
            base = parts[0].lower() if parts[0] in ("Co", "Ch", "Cl", "Cc", "Po", "Ph", "Pl", "Pc") else parts[0]
            resolved = _resolve_catalog(name, self.catalog)
            if resolved is None:
                raise ExprError(f"unknown identifier {name!r}")
        return ("ident", name)


def _resolve_catalog(name: str, catalog: dict[str, Any]) -> Any:
    """Return catalog entry for *name*, or ``None`` if not known.

    The catalog is a flat dict of exposed variable paths (e.g. ``"C.o"``,
    ``"VP1.vah"``, ``"time"``, ``"entry"``). Short candle aliases
    ``Co Ch Cl Cc Po Ph Pl Pc`` and constants ``tick pi bars_from_open``
    are always accepted.
    """
    if name in catalog:
        return catalog[name]
    always = {
        "Co", "Ch", "Cl", "Cc", "Po", "Ph", "Pl", "Pc",
        "tick", "pi", "bars_from_open", "time", "entry",
        "true", "false",
    }
    if name in always:
        return {"kind": "scalar"}
    return None


# ── evaluator ────────────────────────────────────────────────────────────

def _numval(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def _boolval(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    n = _numval(v)
    if n is None:
        return None
    return n != 0.0


def _binop(op: str, a: Any, b: Any) -> Any:
    if op in ("&&", "||"):
        ab, bb = _boolval(a), _boolval(b)
        if op == "&&":
            if ab is False or bb is False:
                return False
            if ab is None or bb is None:
                return None
            return True
        if ab is True or bb is True:
            return True
        if ab is None or bb is None:
            return None
        return False
    if op in ("==", "!="):
        if isinstance(a, str) or isinstance(b, str):
            eq = (str(a) == str(b))
            return eq if op == "==" else not eq
        na, nb = _numval(a), _numval(b)
        if na is None or nb is None:
            return None
        return (na == nb) if op == "==" else (na != nb)
    na, nb = _numval(a), _numval(b)
    if na is None or nb is None:
        return None
    if op == ">":  return na > nb
    if op == "<":  return na < nb
    if op == ">=": return na >= nb
    if op == "<=": return na <= nb
    if op == "+":  return na + nb
    if op == "-":  return na - nb
    if op == "*":  return na * nb
    if op == "/":  return None if nb == 0 else na / nb
    if op == "%":  return None if nb == 0 else na % nb
    raise ExprError(f"internal: unknown binop {op!r}")


def evaluate(ast: Any, bindings: dict[str, Any], helpers: dict[str, Callable[..., Any]] | None = None) -> Any:
    helpers = helpers or {}
    tag = ast[0]
    if tag == "num":  return ast[1]
    if tag == "str":  return ast[1]
    if tag == "bool": return ast[1]
    if tag == "null": return None
    if tag == "ident":
        name = ast[1]
        if name in bindings:
            return bindings[name]
        # dotted lookup fallback: nested dict-style bindings (rare — most are flat)
        parts = name.split(".")
        cur: Any = bindings
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur
    if tag == "u-":
        v = _numval(evaluate(ast[1], bindings, helpers))
        return None if v is None else -v
    if tag == "u!":
        v = _boolval(evaluate(ast[1], bindings, helpers))
        return None if v is None else not v
    if tag == "?:":
        cond = _boolval(evaluate(ast[1], bindings, helpers))
        if cond is None:
            return None
        return evaluate(ast[2] if cond else ast[3], bindings, helpers)
    if tag == "call":
        fname, args = ast[1], ast[2]
        vals = [evaluate(a, bindings, helpers) for a in args]
        return _call(fname, vals, bindings, helpers)
    # binop
    a = evaluate(ast[1], bindings, helpers)
    b = evaluate(ast[2], bindings, helpers)
    return _binop(tag, a, b)


def _call(fname: str, vals: list[Any], bindings: dict[str, Any], helpers: dict[str, Callable[..., Any]]) -> Any:
    if fname == "min":
        nums = [_numval(v) for v in vals]
        nums = [n for n in nums if n is not None]
        return min(nums) if nums else None
    if fname == "max":
        nums = [_numval(v) for v in vals]
        nums = [n for n in nums if n is not None]
        return max(nums) if nums else None
    if fname == "abs":
        n = _numval(vals[0]) if vals else None
        return None if n is None else abs(n)
    if fname == "floor":
        n = _numval(vals[0]) if vals else None
        return None if n is None else math.floor(n)
    if fname == "ceil":
        n = _numval(vals[0]) if vals else None
        return None if n is None else math.ceil(n)
    if fname == "round":
        n = _numval(vals[0]) if vals else None
        if n is None: return None
        digits = int(_numval(vals[1]) or 0) if len(vals) > 1 else 0
        return round(n, digits)
    if fname == "next_up_of":
        if len(vals) < 2:
            return None
        ref = _numval(vals[0])
        if ref is None: return None
        candidates = [_numval(v) for v in vals[1:]]
        above = [c for c in candidates if c is not None and c > ref]
        return min(above) if above else None
    if fname == "next_down_of":
        if len(vals) < 2:
            return None
        ref = _numval(vals[0])
        if ref is None: return None
        candidates = [_numval(v) for v in vals[1:]]
        below = [c for c in candidates if c is not None and c < ref]
        return max(below) if below else None
    fn = helpers.get(fname)
    if fn is None:
        raise ExprError(f"function {fname!r} not available in this evaluation context")
    try:
        return fn(*vals)
    except Exception as exc:  # noqa: BLE001
        raise ExprError(f"call {fname}(...) failed: {exc}") from exc


def compile_expr(src: str, *, catalog: dict[str, Any] | None = None) -> Any:
    """Parse *src* to an AST, checking identifiers against *catalog* when given."""
    if not isinstance(src, str) or not src.strip():
        raise ExprError("empty expression")
    try:
        toks = _tokenize(src)
        return _Parser(toks, catalog).parse()
    except ExprError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExprError(f"parse error: {exc}") from exc
