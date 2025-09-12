from St0_Fns import *
from St1_Fns import *
from St2_Fns import *
from sympy import *
from itertools import combinations_with_replacement
import itertools 
from collections import defaultdict, deque
from itertools import combinations
import sympy as sp
from collections import Counter
'''Assume that input is a GPAC in CRN-implementable form, having arbitrary degree.'''

'''Outputs a population protocol implementable system of ODEs.'''

do_log = False
x0 = Symbol('x_0')
uno = Symbol('x_uno')

class DecompWithBD:

    def __init__(self,input_system, input_iv):
        self.variables = list(input_system.keys()) # The original system variables
        #self.variables.extend([x0,uno])#uno3
        self.crn_system = input_system #Store
        self.crn_iv = input_iv
        self.system = input_system
        self.peaks = {}
        self.uncleaned = None
        self.uncleaned_iv = None
        self.bdsys = None
        self.bdsys_iv = None
        self.TPP = None
        self.vvars = deque() # Contains newly introduced variables as they appear. Each to be processed.

    '''The full conversion of self.input_system to a TPP. 
    
    - limit_sum_est is a user's guess at the limiting sum of the input system, or discovered by stability analysis or approximated via simulation. 
    It is used to determine an adequate initial concentration of the fuel species, x0.'''
    def convert(self,clean=True):
        self.decomp()
        self.post_decomp_bd()
        self.bdsys_iv = self.convertIV()
        self.uncleaned = self.bdsys
        self.uncleaned_iv = self.bdsys_iv
        if clean:
            self.bdsys, self.bdsys_iv = clean_names_map(self.bdsys, {Symbol("x_1") : Symbol("x_1"), x0: x0, uno : uno},self.bdsys_iv)
        self.TPP = self.buildTPP(x0)
        return None
    
    """
    Convert the initial values from self.input_iv (for the original system)
    to a new dictionary of initial values for the compiled system.

    - Original variables retain their initial values.
    - v-variables are initialized as the product of corresponding input_iv powers.
    - All other new/special variables (e.g. x_0, x_uno) are initialized as well.
    """
    def convertIV(self):

        new_iv = {}
        input_iv = self.crn_iv

        for var in self.bdsys.keys():
            if var in input_iv:
                new_iv[var] = input_iv[var]

            elif is_vvar(var):
                powers = v_to_pow(var)
                val = 1
                for base, power in zip(self.variables, powers):
                    val *= input_iv.get(base, 0)**power
                new_iv[var] = val                

        new_iv[uno] = 1
        # This is a crude default estimate based on very little information. A more sophisticated calculation is used in the main compile function.
        # This is likely to 'crash' t0 0 within moments of simulation - and if it crashes, it crashes very quickly. Simply re-initialize the system with a higher x0 value.
        # x0 is an amount of fuel - there exists an amount of fuel which will do the job, any amount that
        # allows x0 to reach equilibrium above 0
        new_iv[x0] = 10 * len(self.bdsys)  
        return new_iv


    """Main decomposition function responsible for transforming input system into a non-homogeneously degree 2 system. 
    Another function - post_decomp_bd() - then converts that into a homogeneously degree 3 TPP-implementable system.
    Note the call to process_v_var, which handles newly introduced variables recursively.

    The recursive approach here is predicated on a specific approach to decomposition. There are a huge family of
    approaches to decomposition that could be implemented instead.
    
    See convert()"""
    def decomp(self):
        for var, expr in self.system.items():
            terms = expand(expr).as_ordered_terms()
            newterms = []
            for term in terms:
                if len(term.free_symbols) == 0: 
                    newterms.append(term)
                    continue
                coeff, term = term.as_coeff_Mul()
                total_degree = sum(degree(term, var) for var in term.free_symbols)
                if (total_degree >2):
                    if coeff > 0: # positive terms
                        v1,v2 = self.group_terms(term)
                        newterms.append(coeff*v1*v2)
                        if not self.handled_or_queued(v1): 
                            self.vvars.append(v1)
                        if not self.handled_or_queued(v2): 
                            self.vvars.append(v2)
                    else: #negative terms
                        rem = term/(var)
                        v = Symbol(mon_to_v(self.variables,rem))
                        newterms.append(coeff*v*var)
                        if not self.handled_or_queued(v): 
                            self.vvars.append(v)
                elif total_degree == 2: #If degree is two, and since we assume the system is already CRN form, don't need to do anything.
                        newterms.append(coeff*term )
                        
                elif total_degree == 1: # If degree is 1, assume form is right - just need to play one instance of the one-trick.
                        newterms.append(coeff*(term))
                
            self.system[var] = sum(newterms)
            if do_log:
                print(f'{var} : {self.system[var]}')

        while(self.vvars):
            v = self.vvars.popleft()
            self.process_vvar(v) # generates terms and adds any new vvars to self.vvars


        return None

    '''Generate the ode for a v-variable and then fit it to homogenous degree 2 form via one-trick and further heuristic decomposition.'''
    def process_vvar(self,vvar):
        varpow = v_to_pow(vvar)
        ode = self.get_ode(varpow)
        newode = 0
        terms = ode.as_ordered_terms()
        for term in terms:
            coeff, term = term.as_coeff_Mul()
            pow = self.absorb_vvars_and_vars_into_pow(term) # The powers of the entire term (combining v-variables and regular variables as necessary)
            total_degree = sum(degree(term, var) for var in term.free_symbols)
            if total_degree > 2:
                if coeff > 0:
                    v1,v2 = self.group_terms(term)
                    if not self.handled_or_queued(v1) and not v1 == vvar:
                        self.vvars.append(v1)
                    if not self.handled_or_queued(v2) and not v2 == vvar:
                        self.vvars.append(v2)
                    newode += coeff*v1*v2
                else:
                    therest = self.powminus(pow, varpow )
                    if sum(therest) == 0:
                        newode += coeff* vvar 
                        continue # the term just equates to the vvar - there's no other stuff to include
                    var = self.vvar_or_var(therest)
                    if not self.handled_or_queued(var) and not var == vvar:
                        self.vvars.append(var)
                    newode += coeff*var * vvar 
            elif total_degree == 2:
                # If the term exactly equals the v-variable itself, rewrite as coeff * vvar
                if self.absorb_vvars_and_vars_into_pow(term) == varpow:
                    newode += coeff * vvar
                else:
                    newode += coeff * term
                    self.queue_new_vvars_from_term(term)
            elif total_degree == 1:
                newode += coeff*term 
            else:
                raise Exception("v-variable seems to have a term of degree 0.")
        self.system[vvar] = newode
        if do_log:
            print(f'{vvar} : {newode}')
    

    '''This approach performs balancing dilation and the one-trick on the system that has been decomposed. Another approach would be to check and 'fix' terms *while* decomposing, and that might be marginally more efficient. This seems less error-prone.'''
    def post_decomp_bd(self):
        # For each variable in the system
        self.bdsys = self.system
        for var in self.system:
            expr = self.system[var]
            new_terms = []
            for term in expand(expr).as_ordered_terms():
                coeff, mono = term.as_coeff_Mul()
                degree_sum = sum(degree(mono, v) for v in mono.free_symbols)
                if degree_sum == 2:
                    new_terms.append(coeff * mono * x0)
                elif degree_sum == 1:
                    new_terms.append(coeff * mono * uno * x0)
                elif degree_sum == 0:
                    new_terms.append(coeff * uno**2 * x0)
                else:
                    # Should not happen, but just in case
                    new_terms.append(term)
            self.bdsys[var] = sum(new_terms)
        self.bdsys[uno] = 0 #uno**2 * x0 
        self.bdsys[x0] = -sum([expr for expr in self.bdsys.values()]) # x0's rate is the negative sum of all other rates
        return None

    # def lambda_trick(self):
    #     lamfrac = .001 # a portion of the lambda reserved for a constant variable
    #     lam = (1-lamfrac)*get_lam(sympify(self.raw_vsystem),[0]*(len(self.raw_vsystem)))#-1) + [1])
    #     self.unoval = lamfrac 
    #     return lam
    
    def queue_new_vvars_from_term(self, term):
        """
        For a given term, extract v-variables and add unprocessed ones to self.vvars.
        """
        termvars = self.get_mon_as_term_powers_list(term)
        for var, _ in termvars:
            if is_vvar(var) and not self.handled_or_queued(var):
                self.vvars.append(var)
                print(var)

    '''Returns true if var has already been processed or is in the queue to be processed, false otherwise.'''
    def handled_or_queued(self,var):
        return var in self.system.keys() or var in self.vvars


    '''Given a SymPy monomial (single term), return a list of (variable, power) pairs.
    Constants are ignored. Returns powers > 0 only.
    Example: 3*v_[a,b,c]**2*x_2 → [(v_[a,b,c], 2), (x_2, 1)]'''
    def get_mon_as_term_powers_list(self,term):
        _, monomial = term.as_coeff_Mul()
        powers_dict = monomial.as_powers_dict()
        return [(base, exp) for base, exp in powers_dict.items() if base.is_Symbol and exp != 0]


    '''Convert the monomial term to a list of powers and then call get_best... to decompose it.
    Returns a pair p1,p2.'''
    def group_terms(self,term):
        pows = self.absorb_vvars_and_vars_into_pow(term) #montopow(term)
        p1,p2 = self.get_best_decomp_2ary(pows) # NEed to enforce that this does not reach above the current max degree v variable
        p1mod = self.vvar_or_var(p1)
        p2mod = self.vvar_or_var(p2)
        return p1mod,p2mod
    
    '''Extract v-variables and treat the rest as one monomial, then sum the power lists together'''
    def absorb_vvars_and_vars_into_pow(self, term):
        vterms = []
        monomial = 1
        
        for factor in term.args if term.is_Mul else [term]:
            if factor.is_Pow and factor.base.is_Symbol and factor.base.name.startswith('v_'):
                if factor.exp != 1:
                    vterms.extend([factor.base] * factor.exp)
                else:
                    vterms.append(factor.base)
            elif factor.is_Symbol and factor.name.startswith('v_'):
                vterms.append(factor)
            else:
                monomial *= factor
        
        # Convert v-variables to power lists
        v_pows = [v_to_pow(v) for v in vterms]
        
        # Convert the remaining monomial to a power list
        mon_pows = mon_to_pow(self.variables, monomial)
        
        # Add all power lists together element-wise
        pows = [sum(x) for x in zip(*v_pows, mon_pows)]
        
        return pows

    '''Interpret a single pow list as either a v-variable or a single original variable'''
    def vvar_or_var(self,pows):
        s = sum(pows)
        if s == 0:
            raise Exception("We have descended too far... pow list with sum 0.")
        elif s == 1:
            return self.getvar(pows)
        else:
            return Symbol(pow_to_v(pows))
        
    '''Assume pows has sum(pows)= 1 and get the variable corresponding to the nonzero index.'''
    def getvar(self,pows):
        for var, pow in zip(self.variables, pows):
            if pow != 0:
                return var
        return None 
    
    def powminus(self,pow1,pow2):
        ret = []
        for i in range(len(pow1)):
            ret.append(pow1[i]-pow2[i])
        return ret


    def is_defined(self, vvar):
        return vvar in self.system or vvar in self.vvars

    """
    Define the ordinary differential equation for v_[pows]/dt.
    """
    def get_ode(self, pows):
        ode_terms = []
        
        for i, power in enumerate(pows):
            if power > 0:
                new_pows = pows.copy()
                new_pows[i] -= 1
                coeff = power
                
                terms = []
                for var, exp in zip(self.variables, new_pows):
                    if exp > 0:
                        terms.append(var**exp)
                
                if power >= 1:
                    var = self.variables[i]
                    terms.append(self.system[var])
                
                term = expand(coeff * Mul(*terms))
                ode_terms.append(term)
        
        ode = Add(*ode_terms)
        return ode
    
    '''Greedily decompose a term (qua list of powers) into a pair of lists of powers. 
    Greedy heuristic: 
        1. (best option) decompose into two terms that have already been defined previously.
        2. (the rest of the time) decompose into two terms that are approximately equal so that we
            jump as low into the lattice towards [0,0,...,0] as possible. 
            
            Explicitly: we *avoid* decomposing in such a way that one variable has been defined and another hasn't,
            because this often uses a defined low-power variable and introduces an undefined high-power variable,
            which is inefficient.
            
            There are situations where one might want to check for a "nearly half" variable already defined. This is 
            marginally more complicated to reason about.'''
    

    # [0,9,0,8] -> [0,7,0,8] (not defined) + [0,2,0,0] (already defined) OR [0,5,0,4] + [0,4,0,4]
    def get_best_decomp_2ary(self, pows):
        best_decomp = None
        best_score = (float('inf'), float('inf'))  # Lower is better

        for i in range(2, 1, -1):  # Only try binary splits
            for decomp in decompose_list_dfs(pows, i):
                max_deg = max(sum(part) for part in decomp)
                undefined_count = sum(
                    not self.is_defined(self.vvar_or_var(part))
                    for part in decomp if sum(part) > 1  # Don't penalize atoms
                )
                score = (max_deg, undefined_count)

                if score < best_score:
                    best_score = score
                    best_decomp = decomp

                if best_score == (sum(pows) // 2, 0):  # Early exit: perfect balanced & all defined
                    return best_decomp

        # If no good option found, default to naive even split
        if best_decomp:
            return best_decomp
        else:
            return self.split_appx_even(pows)

    ''' Split pows into two lists p1 and p2 so that pows = p1 + p2, element-wise, and sum(p1) and sum(p2) are within 1 of each other.'''
    def split_appx_even(self, pows):
        p1 = []
        p2 = []

        for value in pows:
            half_value = value // 2
            p1.append(half_value)
            p2.append(value - half_value)

        return p1, p2


    def buildTPP(self, fuel_var):
            def extract_terms(expr):
                expr = sp.expand(expr)
                return list(expr.args) if isinstance(expr, sp.Add) else [expr]

            def decompose_term(term):
                """Return (coefficient, list of variable names in UPPERCASE, with powers expanded)"""
                term = sp.expand(term)
                if term.is_Number:
                    return term, []

                coeff = 1
                factors = []

                if isinstance(term, sp.Mul):
                    args = term.args
                else:
                    args = [term]

                for arg in args:
                    if arg.is_Number:
                        coeff *= arg
                    elif isinstance(arg, sp.Pow):
                        base, exp = arg.args
                        if not exp.is_Integer or exp < 1:
                            raise ValueError(f"Non-integer or negative exponent in term: {term}")
                        factors.extend([str(base).upper()] * int(exp))
                    elif isinstance(arg, sp.Symbol):
                        factors.append(str(arg).upper())
                    else:
                        raise ValueError(f"Unexpected factor in term: {arg}")

                return abs(coeff), factors

            fuel_str = str(fuel_var).upper()
            reactions = []

            for var, expr in self.bdsys.items():
                var_str = str(var).upper()
                if var_str == fuel_str or var_str == 'X_UNO':
                    continue  # skip x0's own equation

                for term in extract_terms(expr):
                    coeff, factors = decompose_term(term)
                    if fuel_str not in factors or len(factors) != 3:
                        raise ValueError(f'Monomial {term} is not cubic or does not contain the fuel species.')  # must be cubic and contain fuel
                    # Identify the two species besides fuel
                    molecules = [f for f in factors if f != fuel_str]
                    A, B = molecules

                    lhs = [A, B, fuel_str]
                    rhs = []

                    if term.could_extract_minus_sign(): # One of A or B is var_str (otherwise the term can't be negative)
                        C = A if B == var_str else B
                        rhs = [C, fuel_str, fuel_str]
                    else:
                        rhs = [A, B, var_str]

                    reactions.append((coeff, lhs, rhs))
            return reactions
    


''' Decompose a list of integers pows into k, possibly repeated, lists of integers pows1, pows2, ..., powsk such that
 the element-wise sums over these lists equal pows.
 
 Evaluated lazily (yields), because normally we'll find a suitable decomposition 'on the way'.'''
def decompose_list_dfs(pows, k):
    n = len(pows)  # Length of the input list

    # Helper function for DFS
    def dfs(current, remaining, depth):
        if depth == k - 1:
            # If we are at the last list, the remaining part is what we need
            if all(x >= 0 for x in remaining):
                yield current + [remaining[:]]
            return
        
        # Generate valid sublists where each element is <= the corresponding remaining element
        for indices in itertools.product(*(range(remaining[i] + 1) for i in range(n))):
            sub = list(indices)
            next_remaining = [remaining[i] - sub[i] for i in range(n)]
            # Ensure we don't allow negative remaining values
            if all(x >= 0 for x in next_remaining):
                yield from dfs(current + [sub], next_remaining, depth + 1)

    # Filter out any decompositions that are not ordered to avoid duplicates
    for decomposition in dfs([], pows, 0):
        if decomposition == sorted(decomposition):
            yield decomposition

"""
Return the set of 'peak' monomials from a system.
A monomial m1 is a peak if no other monomial m2 in the system satisfies m1 < m2 elementwise.
"""
def peaks(input_sys, variables):
    def term_to_monomial(term):
        powers = [0] * len(variables)
        if isinstance(term, Mul):
            factors = term.args
        else:
            factors = [term]
        for f in factors:
            if f in variables:
                idx = variables.index(f)
                powers[idx] += 1
            elif isinstance(f, sp.Pow):
                base, exp = f.args
                if base in variables:
                    idx = variables.index(base)
                    powers[idx] += int(exp)
        return tuple(powers)

    # Extract all monomials from all expressions
    monomials = set()
    for expr in input_sys.values():
        expr = expand(expr)
        terms = expr.args if isinstance(expr, Add) else [expr]
        for t in terms:
            _, t_expr = (t.as_coeff_Mul() if isinstance(t, Mul) else (1, t))
            mon = term_to_monomial(t_expr)
            monomials.add(mon)

    # Remove dominated monomials
    peaks = set()
    for m in monomials:
        if not any(
            all(mi <= mj for mi, mj in zip(m, m2)) and m != m2
            for m2 in monomials
        ):
            peaks.add(m)

    return peaks


def lattice_union_size(peaks):
    """
    Computes the number of nonzero vectors v such that for at least one peak v <= peak (componentwise).

    Parameters:
        peaks (list of list[int]): Each peak defines a cuboid from 0 to peak (inclusive).

    Returns:
        int: Total number of distinct nonzero vectors in the union of all cuboids.
    """
    total = 0
    n = len(peaks)

    for r in range(1, n + 1):
        for combo in combinations(peaks, r):
            min_vec = [min(v[i] for v in combo) for i in range(len(combo[0]))]
            count = prod(x + 1 for x in min_vec) - 1  # total points in cuboid minus the zero vector
            total += (-1) ** (r + 1) * count

    return total

"""
Check if the given SymPy symbol is a v-variable (i.e., name starts with 'v_[').
"""
def is_vvar(symbol):
    return symbol.is_Symbol and str(symbol).startswith("v_[")

# Define canonical variable symbols
x, y, z = symbols("x y z")

a, b, c = symbols("x_1 x_2 x_3")
# === Helper to compute v-variable initial values === #
def vvar_initial_value(vsym, base_vals, variables):
    powers = v_to_pow(vsym)
    return prod(base_vals[var]**exp for var, exp in zip(variables, powers))

def compute_initial_conditions(system, base_vals: dict,variables):
    init_vals = {}
    for var in system:
        if is_vvar(var):
            init_vals[var] = vvar_initial_value(var, base_vals, variables)
        elif var in base_vals:
            init_vals[var] = base_vals[var]
        else:
            raise Exception(f"Cannot assign initial value to unknown variable {var}")
    return [init_vals[v] for v in system]

