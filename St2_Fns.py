from St0_Fns import *
from St1_Fns import *
from sympy import *
from sympy.utilities.lambdify import *
import numpy as np
import scipy.integrate as intg
from math import (e)
from decimal import Decimal, getcontext, ROUND_DOWN
from re import search


   
'''Apply scipy.integrate.solve_ivp to system of ODE expressions evalf'd on values given as y to sys_rhs.'''

def solve_sys_numeric_solve_ivp(system, iv, interval, steps):
    # Ensure the variable order is consistent
    variables = list(system.keys())

    # Convert symbolic expressions to numerical functions
    funcs = [lambdify(variables, expr, modules="numpy") for expr in system.values()]

    def sys_rhs(t, y):
        # Evaluate each ODE using the current state y
        return [f(*y) for f in funcs]

    # Call the integrator
    sol = intg.solve_ivp(fun=sys_rhs, t_span=interval, y0=iv, max_step=steps)

    return sol


'''Not part of the conversion workflow. '''
def get_limit_sum_est(crn,crn_iv,interval = [0,10]):
     soln = solve_sys_numeric_solve_ivp(crn,list(crn_iv.values()),interval,2) #,np.array([1]* len(system)),interval,2)
     max = get_max(soln)
     return max



'''Get lambda value for lambda trick, based on max value of system solution curve over some span. Look for 
the system to stabilize (or seem to stabilize...).'''
''' Typically, want to use numerical methods to estimate maximum sum of variable values over the course of the execution of system. Use that to determine lambda for lambda trick, and use that to determine c satisfying x1 + lambda * ( x2 + ... + xn) < 1-c. This makes it possible to define an adequate x0 later, with the rescaled variables xi := lambda*xi (i = 2...n).'''
def get_lam(system, iv=None,interval = [0,10] ):
    if(iv == None):
        iv = [0] * len(system)
    max = 0
    attempts = 0
    stabilized = false
    soln = None
    #solve_numeric_scipy(system)
    while not stabilized and attempts < 5:
        soln = solve_sys_numeric_solve_ivp(system,iv,interval,2) #,np.array([1]* len(system)),interval,2)
        max = get_max(soln)
        stabilized, max = is_stabilized(soln, max)
        if(stabilized):
            return get_lam_from_max(max)
        else: 
            iv = soln.y[:, -1] #new initial value is just the end values of the last run. Don't need to re-evaluate all that.
            attempts+=1
            print("System has not stabilized in search for lambda value on {attempts} attempts. Re-attempting with extended evaluation period.")
    # Tried [attempts] number of times to extend the interval and look for stability - did not find it. 
    # System might take a lot longer to stabilize or it might not be bounded
    print("System failed to stabilize over {attempts} attempts. Aborting compilation.")
    return None          

'''Helper subroutine for getting lambda once an adequately stable solution has been found. 
Input: a bound (estimate) for the system var sum'''
def get_lam_from_max(max_sum):
    lam = 1/ (2* max_sum)
    return lam
                

def compute_relative_changes(soln_y):
    if soln_y.ndim == 1:  # If soln_y is 1-dimensional
        all_changes = relative_change(soln_y)
    else:  # If soln_y is 2-dimensional
        num_vars = soln_y.shape[0]
        all_changes = []

        for var_idx in range(num_vars):
            values = soln_y[var_idx, :]
            changes = relative_change(values)
            all_changes.extend(changes)  # Collect changes for all variables

    return all_changes

def relative_change(values):
    changes = []
    for i in range(1, len(values)):
        change = np.abs(values[i] - values[i-1]) / np.maximum(np.abs(values[i-1]), 1e-10)  # Avoid division by zero
        changes.append(change)
    return changes

def safe_ptp(arr):
    if arr.ndim == 1:
        # If the array is 1-dimensional, return the peak-to-peak value directly
        return np.ptp(arr)
    else:
        # Otherwise, apply np.ptp along the specified axis
        return np.ptp(arr, axis=1)

def calculate_tolerance(soln, percentile=90):
    soln_y = np.array(soln.y)  # Ensure it's a numpy array
    num_vars = soln_y.shape[0] if soln_y.ndim > 1 else 1  # Number of variables

    # Calculate baseline tolerance based on the range of values for each variable
    value_ranges = safe_ptp(soln_y)  # Peak-to-peak (max - min) range for each variable
    baseline_tolerance = np.mean(value_ranges) / num_vars

    # Compute relative changes
    rel_changes = compute_relative_changes(soln_y)
    perc_change = np.percentile(rel_changes, percentile)
    
    # Adjust tolerance based on the percentile of changes
    tolerance = baseline_tolerance * perc_change
    
    return tolerance


''' Note that this function gets the individual max values and sums those. It is the sum of variable maxima, not the maximum of sums.'''
def get_max(soln):

    if soln.y.ndim == 1:
        # If the array is 1-dimensional, return the maximum value directly
        return np.max(soln.y)
    else:
        # Otherwise, apply np.max along the specified axis
        return np.sum(np.max(soln.y, axis=1))

def is_stabilized(soln, max=0):
    # Compute the maximum of each variable over all time points
    if soln.y.ndim == 1:
        max_values = np.max(soln.y)
        summed_max_values = max_values
    else:
        max_values = np.max(soln.y, axis=1)
        summed_max_values = np.sum(max_values)
        
    # Calculate tolerance
    tolerance = calculate_tolerance(soln)
    # Check stability
    return check_stability_max_combined(summed_max_values, max, tolerance)


def check_stability_max_combined(summed_max_values, max, tolerance=1e-5):
    # Check if the summed_max_values is close to the max within the given tolerance
    stabilized = np.isclose(summed_max_values, max, atol=tolerance)
    return stabilized, summed_max_values


''' Returns the subset of monomials containing the linear monomials. Simply test their degree.'''
def linear_monomials(monomials):
    return [
        monomial for monomial in monomials 
        if len(monomial.as_powers_dict()) == 1 and 
        next(iter(monomial.as_powers_dict().values())) == 1
    ]    


'''Input: sympified ODE dict. Produces singleton dictionary with new variable x_0 and its corresponding ODE.'''
'''Note that since each variable's ODE is expressible as a degree 3 ODE, so is x0.'''
'''Only used *after* lambda trick has been appled and variables x_1...x_n have been rescaled by a factor of lambda, '''
'''i.e. using this trick beforehand will not work - will not produce a conservative system.'''
def getx0(system):
    return {Symbol("x_0"): -sum(system.values())}


def scale_sys(system, lam):
    scaled_system = {}
    for i, (varname, expression) in enumerate(system.items()):
        # If this is the first variable, leave its expression unchanged
        if i == 0:
            scaled_system[varname] = expression
        # Otherwise, scale the expression by lam
        else:
            scaled_system[varname] = lam * expression
    return scaled_system

'''Is the expression a constant?'''
def my_is_constant(expr):
    return (len(expr.free_symbols) == 0)


'''Perform 'standard' balancing dilation operation (mult. by one power of x0) - see Huang, Huls 2022.'''
def balancing_dilation(x, system, x0):
    # Get P_i and Q_i
    P_i = sum(get_p_terms(x,system))
    Q_i = sum(get_q_terms(x,system,factor_v_out = True))

    # Calculate x_prime for the given variable
    x_prime = expand(P_i - Q_i * x) * x0

    return x_prime


def dynamic_round_down(number):
    if number >= 1:
        return math.floor(number)
    elif number > 0:
        number_str = str(number)
        # Find the index of the first non-zero digit after the decimal point
        first_nonzero_index = len(search("\.(0*)", "5.00060030").group(1))
        rounded_str = number_str[:first_nonzero_index + 1]
        return float(rounded_str)
    else:
        return 0 

'''Do not use on non-CRN-implementable systems or systems with negative values'''
'''The resulting system has initial value [1,0,0,...,0].'''
def stage2_main(in_system,iv=None):
    # Part 1: Introduce a new variable x0 and scale the system to fit it
    lam = get_lam(in_system,iv=iv)
    lam = dynamic_round_down(lam)
    system = scale_sys(in_system, lam)
    result = getx0(system) # Create the x0 variable
    x0 = Symbol("x_0")
    print(f'Scaled system is: {system} with lambda = {lam} and x0 = {result}')


    # Create a new dict to hold the modified system
    modified_system = {}

    # Part 2: Rewrite each pi (quadratic), qi (linear) in x′
    for x in system.keys(): # Original system keys. Excludes x0, which is constructed separately
        eq = system[x] #debug
        px = get_p_terms(x, system)
        qx = get_q_terms(x, system, factor_v_out=True)
        lmpx = set(linear_monomials(px))
        lmqx = set(linear_monomials(qx))
        sumterm = (sum(system.keys()) + x0)
        constantsp = set(filter(my_is_constant, lmpx))
        constantsq = set(filter(my_is_constant, lmqx))

        # Apply the "one"-trick and rewrite linear monomials in px
        for lm in lmpx - constantsp:
            newterm = lm * sumterm # i.e. the one trick relative to the new system
            modified_system[x] = system[x].replace(lm, newterm)

        for const_term in constantsp:
            newterm = const_term * sumterm 
            modified_system[x] = modified_system[x].replace(const_term, newterm) if x in modified_system.keys() else system[x].replace(const_term, newterm)

        # Rewrite constant terms in px and qx
        for const_term in constantsq:
            newterm = const_term * sumterm ** 2
            modified_system[x] = modified_system[x].replace(const_term, newterm) if x in modified_system.keys() else system[x].replace(const_term, newterm)

    #include all the variables that did not arise in the above loop
    missed = filter (lambda x: x not in modified_system.keys() ,system.keys() )
    for y in missed:
        modified_system[y] = system[y]

    # Balancing dilation with x0 on the whole system
    for x in modified_system.keys():
        x_bal = balancing_dilation(x, modified_system, x0)
        result[x] = expand(x_bal)

    return result



''' Some attempts at a priori bounds below. One alternate workflow will be to ask a user: would you like to use a very naive upper bound? It can save some time. Hopefully.'''

'''Trying to get bounds faster via Lyapunov approximations'''
import sympy as sp
import numpy as np


def compute_upper_bound(odes):
    vars = list(odes.keys())
    n = len(vars)
    
    # Define the positive definite matrix P (identity matrix for simplicity)
    P = sp.Matrix(np.eye(n))
    
    # Construct the Lyapunov function V(x) = x^T * P * x
    x = sp.Matrix(vars)
    V = (x.T * P * x)[0]  # Ensure V is a scalar
    
    # Compute the time derivative of V, dV/dt
    dVdt = V.diff(x).dot(sp.Matrix([odes[var] for var in vars]))
    
    # Define the region where the system is bounded (example: norm of x <= some value)
    # This can be derived based on the system's characteristics
    # Define the region where the system is bounded (example: norm of x <= some value)
    bound_region = 10  # Example bound, this should be adjusted based on the system

    # Evaluate V(x) over the region ||x|| <= bound_region
    constraints = [x[i]**2 <= bound_region**2 for i in range(n)]
    max_V = sp.Max(*[V.subs([(x[i], bound_region) for i in range(n)]) for _ in constraints])

    # Compute the upper bound using the maximum eigenvalue of P^-1
    eigenvalues = P.inv().eigenvals()
    lambda_max_inv = max(eigenvalues.keys())

    # The maximum value found for c
    upper_bound = sp.sqrt(lambda_max_inv * max_V)
        
    return upper_bound


