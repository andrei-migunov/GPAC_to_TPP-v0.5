import matplotlib.pyplot as plt
from sympy import degree

def get_term_degree(term, exclude=None):
    """
    Returns the total degree of a term (sum of variable exponents),
    excluding any variable in the 'exclude' set.
    """
    if exclude is None:
        exclude = set()
    return sum(degree(term, v) for v in term.free_symbols if v not in exclude)

def plot_term_degree_histogram(system, exclude_vars=None, title="Term Degree Histogram"):
    """
    Plot a histogram showing the distribution of total degrees of terms in the ODE system.

    Parameters:
        system: dict mapping SymPy symbols to expressions
        exclude_vars: set of variables to ignore in degree (e.g., {x_0, x_uno, x_unoval})
    """
    if exclude_vars is None:
        exclude_vars = set()

    all_degrees = []
    for expr in system.values():
        terms = expr.as_ordered_terms()
        for term in terms:
            if term.is_number:
                continue
            deg = get_term_degree(term, exclude=exclude_vars)
            all_degrees.append(deg)

    # Count and plot
    degree_counts = {d: all_degrees.count(d) for d in sorted(set(all_degrees))}
    plt.bar(degree_counts.keys(), degree_counts.values())
    plt.xlabel("Term Total Degree")
    plt.ylabel("Number of Terms")
    plt.title(title)
    plt.grid(True)
    plt.show()
