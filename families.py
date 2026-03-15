# families.py
# ─────────────────────────────────────────────────────────────────
# Prime family membership predicates and the label vector builder.
# All functions take a single prime integer p as input.
# ─────────────────────────────────────────────────────────────────

from sympy import isprime, factorint
from config import LABEL_NAMES


def is_twin(p):
    return isprime(p - 2) or isprime(p + 2)

def is_sg(p):
    return isprime(2 * p + 1)

def is_safe(p):
    return (p - 1) % 2 == 0 and isprime((p - 1) // 2)

def is_cousin(p):
    return isprime(p + 4) or isprime(p - 4)

def is_sexy(p):
    return isprime(p + 6) or isprime(p - 6)

def is_semiprime(n):
    """True iff n is the product of exactly two primes (with multiplicity).
    Uses trial division for small factors, sympy.factorint for the rest."""
    if n < 4:
        return False
    for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if n % q == 0:
            return isprime(n // q)
    return sum(factorint(n).values()) == 2

def is_chen(p):
    """Chen prime: p+2 is prime or semiprime (Chen 1973)."""
    n = p + 2
    return isprime(n) or is_semiprime(n)

def is_isolated(p):
    """Isolated prime: neither p-2 nor p+2 is prime.
    Label uses forward info for labelling only — the model never sees g^+."""
    return not isprime(p - 2) and not isprime(p + 2)


# Ordered to match LABEL_NAMES in config.py
_LABELERS = [is_twin, is_sg, is_safe, is_cousin, is_sexy, is_chen, is_isolated]

def label_prime(p):
    """Return a 7-element binary list, one entry per prime family."""
    return [int(fn(p)) for fn in _LABELERS]
