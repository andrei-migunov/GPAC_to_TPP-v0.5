from decompose_CRN import *
from St0_Fns import *
from St1_Fns import *
import pickle # serialization
from Tools.plotting_etc import *
from sympy import *
import sys
import os

# try:
#     from ppsim import *
#     ppsim_available = True
# except ImportError as e:
#     ppsim_available = False
#     print("ppsim not installed on the version of Python you're using to run this -- ignoring sim TPP option.")

DEBUG = True


'''Reverse direction - use this to test that final ODE-> PP transformation was correct.'''
def tpp_reactions_to_odes(reactions):
    """
    Convert a list of TPP reactions into a system of ODEs.
    Each reaction is of the form (rate, [A, B, C], [X, Y, Z]).

    Returns: dict mapping sympy.Symbol -> sympy expression for the ODE
    """
    dvars = defaultdict(int)  # Will map variable to sympy expression

    for rate, reactants, products in reactions:
        # Convert names to SymPy symbols
        react_syms = [Symbol(s) for s in reactants]
        prod_syms = [Symbol(s) for s in products]
        all_syms = list(set(react_syms + prod_syms))
        
        # Create monomial term = rate * A * B * C
        monomial = rate
        for r in react_syms:
            monomial *= r

        # For each species, update the ODE based on net change in count
        for s in all_syms:
            delta = prod_syms.count(s) - react_syms.count(s)
            if delta != 0:
                dvars[s] += delta * monomial

    # Simplify the expressions
    for k in dvars:
        dvars[k] = expand(dvars[k])

    return dict(dvars)

def to_ppsim_format(reactions, normalize=False):
    """
    Convert reactions of the form (rate constant, lhs, rhs) into a dictionary:
        { (A,B,C): { (X,Y,Z): rate, ... }, ... }

    If normalize=True, the rate constants are converted into probabilities.
    """

    rule_dict = defaultdict(lambda: defaultdict(float))

    # Collect raw data
    for rate, lhs, rhs in reactions:
        lhs_key = tuple(sorted(lhs))
        rhs_key = tuple(sorted(rhs))
        rule_dict[lhs_key][rhs_key] += rate

    if normalize:
        # Normalize each output distribution
        for lhs_key in rule_dict:
            total = sum(rule_dict[lhs_key].values())
            if total > 0:
                for rhs_key in rule_dict[lhs_key]:
                    rule_dict[lhs_key][rhs_key] /= total

    # Convert defaultdicts to plain dicts for output
    return {
        lhs: dict(rhs_dict)
        for lhs, rhs_dict in rule_dict.items()
    }

"""Pretty-print the reactions in self.TPP in chemical reaction format."""
def to_TPP_reactions_str(TPP):
    if TPP is None:
        return "No reactions to display."
    lines = []
    for rate, lhs, rhs in TPP:
        lhs_str = " + ".join(lhs)
        rhs_str = " + ".join(rhs)
        lines.append(f"{lhs_str} ---> {rhs_str} (rate constant: {rate})")
    return "\n".join(lines)

'''Produces a dictionary that maps lowercase variable names to uppercase ones.'''
def uppercase_variable_mapping(vars):
    var_map = {}
    for var in vars:
        name = str(var)
        var_map[var] = Symbol(name.upper())
    return var_map

class CompileHistory:

    def __init__(self):
        self.input_system = None
        self.input_iv = None
        self.input_mainvar = None
        self.cleaned_input_system= None, 
        self.cleaned_input_iv= None,
        self.cleaned_input_mainvar = None,
        self.input_zeroed_system = None
        self.input_zeroed_iv = None
        self.input_zeroed_mainvar = None
        self.crn = None
        self.crn_iv = None
        self.crn_mainvar = None
        self.peaks = None # The highest 'peak' monomials in the crn system before conversion to TPP-implementable
        self.uncleaned_tpp = None
        self.uncleaned_tpp_iv = None
        self.tpp_impl = None
        self.tpp_impl_iv = None
        self.TPP = None
        #self.TPP_iv = None
        self.ppsim_format = None

    def print(self):
        if self.input_system and self.input_iv:
            print(f'Input system with initial value {self.input_iv}: \n')
            print(format_dict(self.input_system))

        if self.crn and self.crn_iv:
            print(f'CRN system with initial value {self.crn_iv}: \n')
            print(format_dict(self.crn))

        if self.peaks:
            print(f'CRN system peak monomials: {self.peaks} \n')

        if self.tpp_impl and self.tpp_impl_iv:
            print(f'TPP-implementable system with initial value {self.tpp_impl_iv}: \n')
            print(format_dict(self.tpp_impl))

        if self.TPP and self.ppsim_format:
            print(f'Final population protocol {to_TPP_reactions_str(self.TPP)}: \n')
            print(format_dict(self.ppsim_format))
            
    def write(self, filename):
        """
        Writes all attribute values of the object to a text file.
        Handles dictionaries and other types appropriately.
        """
        with open(filename, 'w') as f:
            for attr, value in self.__dict__.items():
                f.write(f"{attr}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
            
            # Finally write the reaction form for readability
            f.write(f'TPP in reaction format: \n' + to_TPP_reactions_str(self.TPP) +"\n")
    
    def verify(self):
        #Verify that the intermediate and final systems are consistent with the input
        backwards =  tpp_reactions_to_odes(self.TPP)
        lower_upper_map = uppercase_variable_mapping(self.tpp_impl.keys())
        for var, expr in self.tpp_impl.items():
            sexpr = expand(sympify(expr)).subs(lower_upper_map)
            if not sexpr == backwards[Symbol(str(var).upper())]:
                raise AttributeError(f'Conversion verification error: TPP qua ODE system does not match earlier TPP-implementable ODE. Please report this error! Thanks!')
        return None
    



        
'''
Input: A bounded general-purpose analog computer G, in the form of a Python dictionary G = {x:x',y:y',...,z:z'} mapping variable names to variable ODEs, 
where lim (t -> infty) x(t) = r, i.e. G computes r via x.

Output: A system ready to be compiled via the standard Stage 1 -> 2 -> 3 (-> 4) process described in (Huang, Huls 2022).

This function 
1. Zeroes the system (modifies the system to start with all-0 initial values).
2. Adjusts the real by a constant to account for change resulting from the zero-ing.

The result is a CRN-implementable system whose variable x_1 computes the number r.
'''
def pre_processing(sys, in_iv, leader):
    
    zsys = zeroed_system(sys,in_iv)

    const_offset = in_iv[leader]
    if const_offset != 0:
    
        #before dual-railing the zeroed system, we need to reintroduce a variable that 
        #goes to the actual desired, original value from the input system.
        zsys = add_const_to_x1(zsys,const_offset)
        zsys = clean_names(zsys,Symbol("x_1"))

    return zsys, {k: 0 for k in zsys.keys()}, Symbol('x_1')

'''Serializes obj. filename must end in .pkl .'''
def cache_obj(obj,filename):
    if os.path.exists(filename):
        print(f"Warning: {filename} already exists and will be overwritten.")
        
    try:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object {filename} of type {type(obj)} cached successfully")
    except Exception as e:
        raise Exception(f"Failed to cache object {filename} of type {type(obj)}. Error: {e}")

def fetch_cached(filename):
    obj = None
    try: 
        obj = unpickle(filename) 
        return obj
    except:
        raise Exception(f"Tried to deserialize an object that probably had not been cached in a previous execution, or simply {filename} does not exist locally.")


''' The main function.

system: a general-purpose analog computer represented as a PIVP represented as a dictionary mapping variable names to those variables' ODEs' expressions.
var: a variable in system.keys() which represents the species converging to the desired real number. This special variable is tracked and this property maintained throughout
compilation.

flags:

    verbose           - If True, prints intermediate system outputs and additional information during compilation.
    conversionchecks  - If True, performs form checks on intermediate systems (e.g., verifies CRN is implementable).
    sim               - List of simulation stages to run. Options: ["INPUT", "CRN", "DECOMP", "TPP"].
    user_limit_sum         - User's estimate of the input system's limiting sum-of-values. Used to estimate fuel species x0. The more precisely one can get a sustainable amount of x0 (such that it does not crash to 0 during simulation), the better ODE solver runtime will be.
    cache_filename    - If provided, caches the CompileHistory object to this file (should end with .pkl).
    filename          - If provided, writes a human-readable summary of the compilation process to this file (should end with .txt).
'''
def compile(system, mainvar, iv, pre_process = False, cache_filename=None, filename=None, checks = False, verbose = False, sim = ["DECOMP"], user_limit_sum = None, simtime = 20):
    #INITIAL SYSTEM 
    ch = CompileHistory()
    ch.input_iv = iv
    ch.input_mainvar = mainvar
    ch.input_system = {k: expand(v) for k, v in system.items()}

    # First verify that system is actually a GPAC
    if not is_valid_gpac_system(system):
        raise TypeError(f'Input system is not a GPAC.')

    ch.cleaned_input_system, ch.cleaned_input_iv = clean_names(system, mainvar, iv)
    ch.cleaned_input_mainvar = Symbol('x_1')
    # Expand all sympy expressions in the system dictionary
    ch.cleaned_input_system = {k: expand(v) for k, v in ch.cleaned_input_system.items()}

    if pre_process:
        print("Converting to system with all-0 initial values...")
        ch.input_zeroed_system, ch.input_zeroed_iv, ch.input_zeroed_mainvar = pre_processing(ch.cleaned_input_system, ch.cleaned_input_iv, ch.cleaned_input_mainvar)

    # if DEBUG and pre_process:
    #     zeroedivs = list(ch.input_zeroed_iv.values())
    #     fsp(ch.input_zeroed_system,zeroedivs,mainvar=ch.input_zeroed_mainvar,time_span=(0,simtime),num_points = 250)

    #CHEMICAL REACTION NETWORK OF ARBITRARY DEGREE
    print("Converting to CRN...")
    crn, crn_iv, leader = smart_dual_rail_optimized(ch.cleaned_input_system, ch.cleaned_input_iv, ch.cleaned_input_mainvar) if not pre_process else smart_dual_rail_optimized(ch.input_zeroed_system, ch.input_zeroed_iv, ch.input_zeroed_mainvar)
    ch.crn = crn
    ch.crn_iv = crn_iv
    ch.crn_mainvar = leader
    ch.peaks = peaks(crn,list(crn.keys()))

    if checks:
        if not crn_implementable(crn): raise ValueError('Internal issue: CRN form system is not CRN implementable (conversion failed)')

    if verbose:
        print(f'CRN translation complete, dual railed system below:')
        print(format_dict(crn))
        print(f'Converting CRN to non-homogeneously degree 2 form via substitution-decomposition...')

    crn, crn_iv = clean_names(crn,leader, crn_iv)

    # TERMOLECULAR POPULATION PROTOCOL-IMPLEMENTABLE SYSTEM
    print("Converting to termolecular system...")
    decomp = DecompWithBD(crn,crn_iv)
    decomp.convert() 
    ch.uncleaned_tpp_iv = decomp.uncleaned_iv
    ch.uncleaned_tpp = decomp.uncleaned
    ch.tpp_impl = decomp.bdsys
    ch.tpp_impl_iv = decomp.bdsys_iv
    ch.TPP = decomp.TPP

    num_crn_vars = len(ch.crn)
    num_tpp_vars = len(ch.tpp_impl)

    # If user provided an estimate of the limiting sum of the input system values then the limiting sum of the CRN is estimated at 3 times that.
    # If not, simulate the CRN system for a brief period and take the sum of maximum observed values over the species.
    # The simulated result is of course more likely to be close - but if the user really has a good guess, they can save some compile time this way.
    #limit_sum_est is a user's belief about the limit of the input GPAC system.
    #we scale it by 2 (dual-rail) and by the growth from the crn system to the tpp system
    #if the user doesn't have a guess, we simulate the crn to estimate equilibrium, 
    limit_sum_est = 2*(num_tpp_vars/num_crn_vars)*user_limit_sum if user_limit_sum else 2*(num_tpp_vars/num_crn_vars)*get_limit_sum_est(ch.crn,ch.crn_iv, interval = [0,5])
    ch.tpp_impl_iv[x0] = limit_sum_est

    if verbose:
        print(to_TPP_reactions_str(decomp.TPP))

    print("Converting TPP to ppsim format...")
    ch.ppsim_format= to_ppsim_format(decomp.TPP)


    if verbose:
        print(ch.ppsim_format)

    print("Saving files...")
    if cache_filename:
        cache_obj(ch, cache_filename)

    if filename:
        ch.write(filename)

    # SIMULATIONS, IF SELECTED
    if sim != []:
        print("Running requested simulations. This can take a long time. ...")
        run_simulations(ch, sim, simtime, DEBUG, verbose)
    print (f'Complete. Returning an object containing the full conversion history.')
    return ch

'''Simulate the intermediate systems as requested by user input.'''
def run_simulations(ch, sim,simtime,debug,verbose):
    
    if "INPUT" in sim:
        if debug or verbose:
            print("Simulating (cleaned variable names) input system...")
        iv = list(ch.cleaned_input_iv.values())
        _, lim = fsp(ch.cleaned_input_system,iv,mainvar=ch.cleaned_input_mainvar,time_span=(0,simtime),num_points = 250)
        if lim and (debug or verbose):
            print(f'(Input) Limiting simulation value of main variable is {lim}.')

    if "CRN" in sim:
        if debug or verbose:
            print("Simulating CRN-implementable (dual-railed) system...")
        __dict__, lim = fsp(ch.crn,list(ch.crn_iv.values()),time_span=(0,simtime),num_points = 250)
        if lim and (debug or verbose):
            print(f'(CRN) Limiting simulation value of main variable is {lim}.')

    if "DECOMP" in sim:
        if debug:
            print("Simulating TPP-implementable system (qua deterministic system)...")
        _, lim = fsp(ch.tpp_impl,list(ch.tpp_impl_iv.values()),time_span=(0,simtime),num_points = 250)
        if lim and (debug or verbose):
            print(f'(TPP-implementable) Limiting simulation value of main variable is {lim}.')

    if "TPP" in sim:
        return None
        #TODO
 
"""
Reads a .txt file describing a system, initial values, and optional flags,
then compiles using the main compile(...) function. Users should have *some* freedom, but not a lot.
See the example inputs.txt file for format.
There may be a lot of flags - we should either decide what they are, or have a flexible way to handle adding more.
"""
def compile_from_file(input_filename):


    def read_next_dict(lines, start_idx):
        """Read a dictionary block enclosed by matching { ... } braces."""
        block_lines = []
        brace_count = 0
        started = False
        i = start_idx

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            if "{" in line:
                brace_count += line.count("{")
                started = True
            if "}" in line:
                brace_count -= line.count("}")

            if started:
                block_lines.append(line)
            if started and brace_count == 0:
                break
            i += 1

        if brace_count != 0:
            raise ValueError("Unbalanced braces in dictionary block.")

        return "\n".join(block_lines), i + 1

    # Read all lines first
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # Parse system dictionary
    system_str, next_idx = read_next_dict(lines, 0)
    try:
        system = sympify(ast.literal_eval(system_str))
    except Exception as e:
        raise ValueError(f"Could not parse system dictionary: {e}")

    # Parse IV dictionary
    iv_str, next_idx = read_next_dict(lines, next_idx)
    try:
        iv = sympify(ast.literal_eval(iv_str))
    except Exception as e:
        raise ValueError(f"Could not parse initial values: {e}")

    # Remaining lines: flags
    kwargs = {}
    for line in lines[next_idx:]:
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                val = ast.literal_eval(val)
            except:
                pass
            kwargs[key] = val

    # Identify primary variable
    primary_var = Symbol(kwargs.pop("primary_var")) if "primary_var" in kwargs else Symbol(list(system.keys())[0])

    base_filename = input_filename.split(".")[0]

    return compile(system=system, mainvar=primary_var, iv=iv, cache_filename = base_filename+".pkl",  filename = base_filename + "output.txt", checks = kwargs["checks"], verbose = kwargs["verbose"], sim = kwargs["sim"] )

if __name__ == '__main__':
    compile_from_file(sys.argv[1])

