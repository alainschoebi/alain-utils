# SymPy
try:
    import sympy as sp
except ImportError:
    pass

# Utils
from ..decorators import requires_package

@requires_package("sympy")
def quadratic_polynomial_Abc(
        expr: sp.Expr, vars: list[sp.Symbol] | tuple[sp.Symbol],
    ) -> tuple[sp.Matrix, sp.Matrix, sp.Expr]:
    """
    Given a symbolic quadratic expression depending on `n` variables, this
    function computes the corresponding quadratic form:
       `1/2 x^T A x + b^T x + c`,
    with matrix `A`, vector `b` and scalar `c`.

    Inputs
    - expr:  symbolic expression of the quadratic polynomial
    - vars:  list or tuple of `n` variables (symbols) on which the polynomial
             depends, e.g. [x_1, x_2, ..., x_n].

    Returns
    - A:     (n, n) matrix of the quadratic form
    - b:     (n, 1) vector of the linear form
    - c:     scalar constant term

    Note: the expression can also be parametrized by other symbolic variables
          that are not included in the polynomial variables `vars`.
    """
    # Build polynomial
    poly = sp.Poly(expr, *vars)
    n = len(vars)

    # Check if the polynomial is quadratic
    if not max(poly.degree_list()) == 2:
        raise ValueError(
            "The polynomial is not quadratic. The degree is " +
            f"{poly.degree()} instead of 2."
        )

    A = sp.Matrix.zeros(n, n) # (n, n)
    for i in range(n):
        for j in range(i+1):
            coeff = poly.coeff_monomial(vars[i] * vars[j])
            A[i, j] += coeff # Note: no 1/2 since we 1/2 Ax^2
            A[j, i] += coeff

    b = sp.Matrix.zeros(n, 1) # (n, 1)
    for i, xi in enumerate(vars):
        b[i] = poly.coeff_monomial(xi)

    c = poly.coeff_monomial(1)

    return A, b, c

