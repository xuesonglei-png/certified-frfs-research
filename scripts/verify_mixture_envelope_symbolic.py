"""Exact rational SymPy check of the strict witness-mixture example."""
from sympy import Rational, Symbol, Abs, Max, Min, simplify


def main():
    t=Symbol('t', real=True)
    # Query t in [0,1], full witnesses at 0 and 1, reduced enemy at 1/2.
    single_left=Max(Abs(t),Abs(t-1))-Rational(1,5)*Abs(t-Rational(1,2))
    single_right=single_left
    mixture=Rational(1,2)*Abs(t)+Rational(1,2)*Abs(t-1)-Rational(1,5)*Abs(t-Rational(1,2))
    # Piecewise linear maxima occur at 0, 1/2, and 1.
    points=(0,Rational(1,2),1)
    single_sup=max(simplify(single_left.subs(t,p)) for p in points)
    mixture_sup=max(simplify(mixture.subs(t,p)) for p in points)
    assert single_sup==Rational(9,10) and mixture_sup==Rational(1,2)
    assert mixture_sup<single_sup
    out={'single_upper_bound':str(single_sup),'mixture_upper_bound':str(mixture_sup),
         'strict_gap':str(single_sup-mixture_sup),'exact':True,
         'scope':'symbolic_example_only_not_full_certificate'}
    print(out)


if __name__=='__main__':main()
