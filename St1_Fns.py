from sympy import symbols, sympify, expand, collect
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.core.symbol import Symbol
from sympy import diff
from St0_Fns import *

'''Helper for set_v_variable_ode.

We want to avoid making reference to variables like v_01, which are just aliases
for variables we already have. Instead, replace these with the old variable names.

ASSUME old variable names are x_1,...,x_n. 
Generate the ODE based on the input powers and substitute the derivatives.

:param system: The system dictionary containing expressions for derivatives.
:param pows: The list of powers (e.g., [0,1,0,1] or [0,2,0]).
:return: The ODE with substituted derivatives.
'''

def pows_to_old_vars(system, pows):
    if not any(pows):  # If v_[0,0,0...,0]
        return sympify(0) 

    non_zero_indices = [i for i, p in enumerate(pows) if p > 0]
    
    ode = sympify(0)
    if len(non_zero_indices) == 1 and pows[non_zero_indices[0]] == 2:
        i = non_zero_indices[0]
        x_i = Symbol(f"x_{i+1}")
        x_i_prime = system[x_i]
        ode += 2 * x_i * x_i_prime
        
    elif len(non_zero_indices) == 2:
        i, j = non_zero_indices
        x_i = Symbol(f"x_{i+1}")
        x_j = Symbol(f"x_{j+1}")
        x_i_prime = system[x_i]
        x_j_prime = system[x_j]
        ode += x_i * x_j_prime + x_j * x_i_prime

    return ode





"""
Given a v variable as a string, compute the corresponding ODE
Example
Input: system ; "v_[3,1]"
Output: system satisfying system[Symbol("v_[3,1]")]= Sympify("3v_[2,1] + v_[3,0])"

The function also checks whether v-variables created inside an ODE exist in the system already. 
If not, they need to be added and their ODEs computed also.

:param v_variable: list of integers
:return: None
"""
def set_v_variable_ode(system, v_variable):

    v_pows = v_to_pow(v_variable)

    if (not Symbol(v_variable) in system.keys()) and sum(v_pows) >= 2 :
        #We don't need ODEs for the lowest level variables. Those are just the input variables x_1,x_2,...,x_n
        if sum(v_pows) == 2:
            #specific update for this case to reuse old vars. This is why we don't have to do anything in the case <= 1
            system[Symbol(v_variable)] = pows_to_old_vars(system,v_pows)
             
        else:
            ode = sympify(0)
            for i,pow in enumerate(v_pows):
                if pow > 0:
                    term = Symbol(pow_to_v(decr_at_i(i,v_pows)))
                    if term in system.keys(): # If the term is already in the system, just add it to the ODE
                        ode += pow * Symbol(pow_to_v(decr_at_i(i,v_pows))) * system[Symbol(f"x_{i+1}")]
                    else: #If it's not already in the system, repeat the process for that term, and then add it to the present ODE
                            # We also need to be sure that this term is not a bottom level term
                        set_v_variable_ode(system, pow_to_v(decr_at_i(i,v_pows)))
                        ode += pow * Symbol(pow_to_v(decr_at_i(i,v_pows))) * system[Symbol(f"x_{i+1}")]

            system[Symbol(v_variable)] = ode
         
def get_args_set(expr):
    if isinstance(expr, Add):
        return set(expr.args)
    elif isinstance(expr, Mul):
        return {expr}
    else:
        raise Exception("Expected either Add or Mul type expression.")


'''Get all the monomials that appear anywhere in the ODE system, except
for the ones that are constant values or multiples of a single variable'''
def get_all_monoms(vars, sys):
    monoms = set()
    for expr in sys.values():
        simp = expand(expr)
        args_set = get_args_set(simp)
        for arg in args_set:
            sumpows = sum(mon_to_pow(vars, arg))  # am offline right now, do this better later
            if sumpows > 1:  # check if the total power of the monomial is > 1.
                monoms.add(arg)
    
    return monoms


def get_all_monoms_constants_expr(expr): # can probably do this with a sift
    monoms = set()
    constants = set()
    simp = expand(expr)

    # this needs to be more complicated
    # need to check if the expression expr is a product. if so, then its args represent the terms of a product.
    # if it's a sum, its terms represent terms of a sum. so they have different meanings in those constants.

    for arg in simp.args:
        if len(arg.free_symbols) >0: # having at least one free symbol in the expression means it is not a constant
            monoms.add(arg)
        else:
            constants.add(arg) 
    return monoms, constants

def get_all_monoms_constants_term(term):
    return


''' Takes a sympy expression as input. Specifically, it should be given one term which is itself a product of some expressions.
'''

"""
Given a list of integers representing powers, this function will return the corresponding v_variable as a string

Example
Input: [0,3,4,5]
Output: "v_[0,3,4,5]"

:param pows: list of integers
:return: string
"""
def pow_to_v(pows):
    return "v_" + str(pows).replace(" ", "")

def mon_to_pow(vars,monom):
    powers = [monom.as_coeff_exponent(v)[1] for v in vars]
    return powers

def mon_to_v(vars,monomial):
    #Get the powers of the monomial
    powers  = mon_to_pow(vars, monomial)
    #Create the string literal v_pow e.g. "v_[1,3,0]"
    v_variable = pow_to_v(powers)
    return v_variable

'''Gets the coefficient in front of a monomial'''
def get_coeff(monom):
    return monom.as_coeff_Mul()[0] 

def mon_to_coeff_and_pows(vars, monom):
    powers = mon_to_pow(vars, monom)
    coeff = get_coeff(monom)
    return coeff, powers

'''Parses out a v-variable into a list of powers. Assumes correct representation with as many indices as variables in input system.'''
def v_to_pow(vvar):
    return [int(x) for x in str(vvar)[2:].strip('][').split(',')]

'''Return a list of integers x equal to pows except at index i where x[i] = pows[i]-1'''
def decr_at_i(i,pows):
    x = list(pows)
    x[i] -= 1
    return x


"""
Given an  ode system as a dictionary of strings or ode system as a dictionary of sympy expr,
convert the system down to degree two/one, using v variable substitution
Example
Input: {"x":"-2*(y**3)","y":"-3*(x**2)"}
#Missing original variables
Output: {v_03: 3*v_02, v_02: 2*v_01, v_01: v_00, v_00: 0, v_20: 2*v_10, v_10: v_00}

:param sys: dictionary of strings or dictionary of sympy expr
:return: dictionary of sympy expr
"""
def stage1_main(sys):

    sys = clean_names(sys,"x_1")
    #Get list of variables from system
    variables = sys.keys()
    #Get all the monomials from the system except for the ones that are constant values or multiples of a single variable
    monomials = get_all_monoms(variables,sys)
    subs_dict = {}

    #Transform monomials into v-variables that will be the keys of the new system
    v_variables = set()

    for monomial in monomials:
        v_variable = mon_to_v(variables, monomial)
        coeff, _ = monomial.as_coeff_Mul()
        v_variables.add(v_variable)
        subs_dict[monomial/coeff] = Symbol(v_variable)
        
    v_system = {}
    # Perform substitutions of new v_variables on expressions of old variables
    # As long as we do not include any variables we don't want in sub_dict 
    # (for example, v_010 and v_000) this works out fine.
    for var in sys.keys():
        v_system[var] = sys[var].subs(subs_dict)

    #For each v variable, get its corresponding ode to add to the system

    for current_v_variable in v_variables:
        set_v_variable_ode(v_system, current_v_variable)

    return v_system

'''Alternate stage 1 flow using successive squaring method.'''
# def stage1_alt():
