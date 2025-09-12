from sympy import symbols, pprint,Eq, parsing, Add, Symbol, sift, sympify, div, simplify, expand, srepr, Abs, Pow, Mul, Derivative, Basic
import math # used for GCD calculation
import ast # used to extract a list out of a string
import re

''' Convert a dictionary of variable name : sympy ODE pairs to a dictionary of str: str expressions of these'''
def sym_to_str_dict(sys):
    return {str(var) : (str(sys[var]) if sys[var] != None else None) for var in sys.keys()}

def vvar_to_sympy(vvar):
    return Symbol(f"v_{'_'.join(map(str, vvar))}")

def is_valid_gpac_system(system):
    """
    Verify whether the given input defines a valid GPAC (polynomial, autonomous, first-order) system.

    Parameters:
        system (dict): A dictionary of the form {x: expr_x, y: expr_y, ...}
                       where keys are sympy.Symbol and values are sympy.Expr

    Returns:
        (bool, list): A tuple (is_valid, list_of_errors)
    """

    errors = []

    if not isinstance(system, dict):
        return False, ["Input must be a dictionary mapping sympy.Symbol to sympy.Expr"]

    if not all(isinstance(k, Symbol) for k in system):
        errors.append("All keys must be sympy.Symbols")

    if not all(isinstance(v, Basic) for v in system.values()):
        errors.append("All values must be sympy expressions")

    declared_vars = set(system.keys())

    for var, expr in system.items():
        # Check for undeclared variables
        free_vars = expr.free_symbols
        undeclared = free_vars - declared_vars
        if undeclared:
            errors.append(f"Expression for {var} contains undeclared symbols: {undeclared}")

        # Check for polynomial
        if not expr.is_polynomial(*declared_vars):
            errors.append(f"Expression for {var} is not a polynomial in the declared variables")

        # Check for derivative subexpressions
        if any(isinstance(subexpr, Derivative) for subexpr in expr.atoms(Derivative)):
            errors.append(f"Expression for {var} contains derivatives (must be first-order ODE)")

    return len(errors) == 0, errors


def str_to_sym(sys):

    if(isinstance(sys,str)):
        # Manually parse the string representation of a dictionary
        # Assuming sys is a well-formed dictionary string
        sys = sys.strip("{}")
        sys_dict = {}
        for item in sys.split(','):
            key, value = item.split(':')
            key = key.strip(" '\"")
            value = value.strip(" '\"")
            sys_dict[key] = value
        sys = sys_dict 
        return strdict_to_sym(sys)
    else:
        return strdict_to_sym(sys)

'''Helper - Convert a specifically str(var):str(ODE) dict to a sympy : sympy dict'''
def strdict_to_sym(sys):
    if(isinstance(sys,dict)):
        sym_odes = {}
        var_symbols = {}
        for var in sys.keys():
            var_symbols[var] = symbols(var) #store the variable name/symbol in list of variable symbols
            #sympy has built in parser at least good enough for simple stuff
            eq = parsing.sympy_parser.parse_expr(sys[var])
            sym_odes[var_symbols[var]] = eq
        return sym_odes
    else:
        raise ValueError("Input is not a dictionary.")


'''Convert a system and initial value to an equivalent system with 0 initial values.'''
''' See Xiang thesis pg. 35 .'''
def zeroed_system(sys, iv):
    iv_offset = {var: sympify(var) + sympify(iv[var]) for var in sys.keys()}
    result = {var: eq for var, eq in sys.items()}
    for var in result:
        for var_sub, value_sub in iv_offset.items():
            result[var] = expand(result[var].subs(var_sub, value_sub))
    return result

'''Assume all variables are named consistently x_1 : f1 ,...,x_n: fn. 
Returns a new system x_1 : None, x_2: None, ..., x_k : None, x_{k+1} : f1, ..., x_{n+k} : fn. 
i.e. "scoots" all the variables over to the right by k.
Typically used to make room for a new x_1 variable when preserving a real number computation.
'''
def scootr(sys, k):
    # Create a new dictionary with None values for the first k keys
    new_sys = {f'x_{i+1}': None for i in range(k)}
    
    # Add the original system to the new dictionary, shifting the keys by k
    new_sys.update({f'x_{i+k+1}': eq for i, (var, eq) in enumerate(sys.items())})
    
    # Substitute the old variables with the new ones in the equations. Note we go in reverse
    # so that we're not re-re-re...-resubstituting stuff.
    for i, (var, eq) in enumerate(new_sys.items()):
        if eq is not None:
            for j in reversed(range(0, len(new_sys)+1)):
                old_var = symbols(f'x_{j}')
                new_var = symbols(f'x_{j+k}')
                new_sys[var] = sympify(new_sys[var]).subs(old_var, new_var)
    
    return new_sys

'''If sys.x_1 approaches alpha in the limit, then returns sys2 satisfying
sys2.x_1 -> alpha + c

ASSUME c is a rational number p/q
ASSUME user has done a clean_names before calling, so that x_1 actually exists and is the relevant variable.

Introduces two auxiliary variables: constant c, and x_1 = sys.x_1 + c

Use this function to recover the real number lost when zeroing a system with nonzero IV.
For example, if x1 -> alpha, and x1* -> alpha - 1/2, then add_const_to_x1(sys, 1/2) will have variable x_1 with x1 -> lim(x1* + 1/2) = alpha.
'''
def add_const_to_x1(sys,c):
    q = {"q": f'-q + {c}'} # lim q = c
    newsys = scootr(sys,1) | q | {"x_1" : 'x_2 + q - x_1'} # CRN for addition 
    return sympify(newsys)

'''Given input system sys and a special variable inside sys, mainvar:
Rename all the variables in the system (and update all equations accordingly) so that they have standardized names x_1...x_n,
with x_1 = mainvar (usually, the variable that tracks the real number being computed or property being preserved.
'''
'''TODO: Change mainvar to a list of variables and have clean_names return a function mapping the old variable names to the new ones'''

def clean_names(sys, mainvar, iv = None):
    sys = sympify(sys)
    mainvar = sympify(mainvar)

    # Create a mapping from old variable names to new ones
    var_mapping = {mainvar: Symbol('x_1')}
    i = 2
    for var in sys.keys():
        if var != mainvar:
            var_mapping[var] = Symbol(f'x_{i}')
            i += 1

    # Sort the keys in var_mapping in reverse order
    sorted_keys = sorted(var_mapping.keys(), key=str, reverse=True)

    # Build new_sys with x_1 first, then the rest in original order
    new_sys = {}
    # Add x_1 (mainvar) first
    new_sys[Symbol('x_1')] = sympify(sys[mainvar]).subs({old_var: var_mapping[old_var] for old_var in sorted_keys})
    # Add the rest
    for var in sys.keys():
        if var != mainvar:
            new_eq = sympify(sys[var])
            for old_var in sorted_keys:
                new_eq = new_eq.subs(old_var, var_mapping[old_var])
            new_sys[var_mapping[var]] = new_eq

    # If iv is provided, update its keys to match the new variable names
    if iv is not None:
        # Map IVs to new variable names, then sort by subscript index (x_1, x_2, ...)
        new_iv = {var_mapping[k]: v for k, v in iv.items()}
        new_iv = dict(sorted(new_iv.items(), key=lambda item: subscript_index(item[0])))
        return new_sys, new_iv
    return new_sys

'''Return the vraiable name subscript e.g. 5 from x_5.'''
def subscript_index(sym):
    name = str(sym)
    if "_" in name:
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return float('inf')
    
'''Same thing, but instead of mainvar provide a general mapping of variables to their new names'''
# Right now this isn't updated to sort by variable name before returning like the above function is, but 
# this only really matters early in compilation anyway, and I will fix this later.
def clean_names_map(sys, var_mapping, iv = None, expand_eqs=False):
    sys = sympify(sys)  # Just in case

    # Create a new dictionary to store the renamed system
    new_sys = {}

    # Initialize a counter for the variable names that are not in var_mapping
    i = 1
    for var in sys.keys():
        if var not in var_mapping:
            while Symbol(f'x_{i}') in var_mapping.values():
                i += 1
            var_mapping[var] = Symbol(f'x_{i}')
            i += 1

    # Sort the keys in var_mapping in reverse order
    sorted_keys = sorted(var_mapping.keys(), key=str, reverse=True)

    # Substitute the new variables for the old ones in each equation
    for var, eq in sys.items():
        new_eq = sympify(eq)

        # Substitute the variables in the sorted order
        for old_var in sorted_keys:
            new_eq = new_eq.subs(old_var, var_mapping[old_var])

        new_sys[var_mapping[var]] = new_eq if not expand_eqs else expand(new_eq)

    # If iv is provided, update its keys to match the new variable names
    if iv is not None:
        new_iv = {var_mapping[k]: v for k, v in iv.items()}
        return new_sys, new_iv
    return new_sys

class Graph:
    def __init__(self):
        self.adj = {}  # Adjacency list representation of the graph
        self.vertices = {}  # Maps vertex names (strings) to their indices
        self.V = 0  # Number of vertices in the graph

    def add_vertex(self, name):
        if name not in self.vertices:
            self.vertices[name] = self.V
            self.adj[self.V] = []
            self.V += 1

    def add_edge(self, v, w):
        if v not in self.vertices:
            self.add_vertex(v)
        if w not in self.vertices:
            self.add_vertex(w)
        self.adj[self.vertices[v]].append(self.vertices[w])

    def __str__(self):
        output = "Graph:\n"
        for vertex, index in self.vertices.items():
            output += f"Vertex {vertex}: Edges -> {' '.join([self.get_vertex_name(i) for i in self.adj[index]])}\n"
        return output

    def __repr__(self):
        return f"Graph({self.V}, {self.adj})"

    def tarjan_util(self, u, low, disc, stack_member, st, sccs):
        '''A recursive function to find strongly connected components'''
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stack_member[u] = True
        st.append(u)

        for v in self.adj[u]:
            if disc[v] == -1:
                self.tarjan_util(v, low, disc, stack_member, st, sccs)
                low[u] = min(low[u], low[v])
            elif stack_member[v]:
                low[u] = min(low[u], disc[v])

        w = -1
        if low[u] == disc[u]:
            scc = []
            while w != u:
                w = st.pop()
                scc.append(self.get_vertex_name(w))
                stack_member[w] = False
            sccs.append(scc)

    def tarjan_scc(self):
        disc = [-1] * self.V
        low = [-1] * self.V
        stack_member = [False] * self.V
        st = []
        sccs = []
        
        self.Time = 0

        for i in range(self.V):
            if disc[i] == -1:
                self.tarjan_util(i, low, disc, stack_member, st, sccs)

        return sccs

    def get_vertex_name(self, index):
        for vertex, idx in self.vertices.items():
            if idx == index:
                return vertex


#y affects x if x depends on y
def affects(y,x,sys):
    return depends(x,y,sys)

#Returns True if  y appears in the positive terms of x
def depends(x,y,sys):
    #x and y are sympy expressions
    # check if y appears in yx as a positive term
    for term in get_p_terms(x,sys):
        if term.has(Symbol(y)):
            return True
    return False

#V' = P - QV
#P is a polynomial with positive terms that may or may not
#Q is a polynomial with positive terms that must include V
#   or to put it another way, all the negative terms in v' must include v.

'''Return a list of the terms that are added in the equation.'''
def get_all_terms(v,eq):
    strv = str(v)
    symv = Symbol(strv)
    args = Add.make_args(expand(eq))
    #conditional in case someone's passing a sympyfied dictionary in
    return list(args)

#Get the positive monomials (which sum to P) of the expression 
#note that these monomials may have V in them- e.g. when V -> 2V is a reaction or Y+V -> 2Y+2V or etc.
#Extra lines here handle user passing in string dict instead of sympy. No big deal.
def get_p_terms(v,sys):
    strv = str(v)
    symv = Symbol(strv)
    is_pos = lambda x: x.as_coeff_Mul()[0].is_positive
    #conditional in case someone's passing a sympyfied dictionary in
    args = Add.make_args(expand(sys[v] if v in sys else sys[strv]))
    return sift(args,is_pos,binary=True)[0]

#Get all the negative monomials (sum them and negate them to get the polynomial Q)
#Note that this gets the terms of Qv - it has not factored it out into Q yet.
#Extra lines here handle user passing in string dict instead of sympy. No big deal.
def get_q_terms(v,sys, factor_v_out = False):
    strv = str(v)
    symv = Symbol(strv)
    is_neg_has_v = lambda x: x.as_coeff_Mul()[0].is_negative and x.has(symv)
    args = Add.make_args(expand(sys[v] if v in sys else sys[strv]))
    sifted = sift(args,is_neg_has_v,binary=True)[0]
    if(not factor_v_out):
        return sifted
    else:
        qsansv = [div(term,symv)[0] for term in sifted]
        return qsansv



#Returns all the terms in the ODE for v', which are negative but do not contain v
#Extra lines here handle user passing in string dict instead of sympy. No big deal.
def get_ill_formed(v,sys):
    strv = str(v)
    symv = Symbol(strv)
    #check if there is any negative term which does not contain v
    is_neg_no_v = lambda x: x.as_coeff_Mul()[0].is_negative and not x.has(symv)
    args = Add.make_args(sys[v] if v in sys else sys[str(v)])
    return sift(args,is_neg_no_v,binary=True)[0]


#Boolean version of the above
def has_ill_formed(v,sys):
    has_ill = lambda x: x.as_coeff_Mul()[0].is_negative and not x.has(Symbol(v))
    return has_ill(sys[v] if v in sys else sys[str(v)])

#return True if system is not CRN implementable
def is_ill_formed_system(sys):
    for v in sys.keys():
        if has_ill_formed(v,sys):
            return True
    return False

#Does the system satisfy y'=p-qy for every ODE?
def crn_implementable(sys, outputs = False):
    res = True
    for x in sys.keys():
        bads = get_ill_formed(x,sys)
        if(outputs):
            pprint(sys[x])
            print(f"p terms for {x}': \n {get_p_terms(x,sys)}\n\n")
            print(f"q terms for {x}': \n {get_q_terms(x,sys)}\n\n")
            print(f"bad terms for {x}': \n {bads}\n\n")
        if bads:
            res = False #fails form check
    return res

#For each depednency between variables in the system, add an edge to the graph
def gen_dependencies(sys): 
    G = Graph()
    for x in sys.keys():
        for y in sys.keys():
            if(y != x and depends(x,y, sys)):
                G.add_edge(x,y)
    return G
    
#Smart dual rail: To convert ODE to GPAC using dual-rail approach,
#but selectively: do not dual-rail all variables, just the ones that need that.
def smart_dual_rail_optimized(system, iv, mainVar=Symbol("x_1")):
    """
    Selectively dual-rails the input system, rewriting only those variables that are infected by ill-formed variables.
    Returns a modified initial value for the new system, based on the initial values of the input system.
    
    Steps:
      1. Identify ill equations that need to be dual railed. An equation is ill if any term in 
         its funtion starts with a negative sign and doesnt contain the variable itself
      2. Build a dependency dictionary by going through each expression
      3. Set the ill status based on dependencies
      4. Split the system into ill and good equations
      5. For the ill equations perform a slightly modified naive dual rail (using only the Klinge‐style subtraction method)
            - Duplicate each equation into u and v copies
            - Substitute every variable with (u_ – v_)
            - Expand and split each function into its positive and negative parts
            - Change the u and v equations by subtracting the correction term from the naive method(u_x * v_x * (sum(positive terms) - sum(negative terms)))
      6. For all the good equations substitute every ill varible with (u_ – v_) with regual expressions
      7. Combind both the good and ill system and return it
    """
    
    leader = mainVar # The variable that tracks the number computation might change during this process.
    # Make sure the system is all sympified
    system = sympify(system) #{k: sympify(v) for k, v in system.items()}
    
    ### Step 1. Identify ill keys: keys thats equations have a term that begins with '-' and does not contain the key
    ill_keys = set()
    for key, expr in system.items():
        for term in expr.as_ordered_terms():
            # Check if the string has a term starts with '-' and the term does not involve the key
            if str(term).startswith("-") and (not term.has(key)):
                ill_keys.add(key)
                break

    ### Step 2. Build dependency dictionary:
    # Remove spaces and change '-' to '+-')
    # and collect positive dependency variables (things that are x_? that doesnt contain the key itself)
    dep_pattern = re.compile(r'x_\w+')
    depends = {}
    for key, expr in system.items():
        expr_str = str(expr).replace(" ", "").replace("-", "+-")
        terms = [term for term in expr_str.split("+") if term]
        pos_vars = set()
        for term in terms:
            if term.startswith("-") or str(key) in term:
                continue
            pos_vars.update(dep_pattern.findall(term))
        depends[key] = pos_vars

    ### Step 3. get ill status using dependencies:
    # Set each key to False except it is already ill.
    need_dual = {key: (key in ill_keys) for key in system.keys()}
    # Go through everything until it has a time nothing changes:
    changed = True
    while changed:
        changed = False
        for key, dep_vars in depends.items():
            if not need_dual[key]:
                # Check if the dependancys of everything have anything that is ill
                if any(var in {str(ill_key) for ill_key in ill_keys} for var in dep_vars): #returns true if any of the dependancys and in the ill keys
                    need_dual[key] = True
                    ill_keys.add(key)
                    changed = True

    ### Step 4. Split system into ill and good parts:
    ill_sys = {k: v for k, v in system.items() if need_dual[k]}
    good_sys = {k: v for k, v in system.items() if not need_dual[k]}

    ### Step 5. Dual rail the ill equations:
    # Create new ones for new u and v variables (for every key in ill_sys)
    uvars = {key: symbols(f"u_{str(key)}") for key in ill_sys.keys()}
    vvars = {key: symbols(f"v_{str(key)}") for key in ill_sys.keys()}
    
    dualed = {}  # new system for ill equations
    
    # check if mainVar is part of the ill system and if it is add extra equations for z_h and z
    if mainVar in ill_sys:
        u_main = uvars[mainVar]
        v_main = vvars[mainVar]
        dualed[Symbol("z_h")] = 1 - Symbol("z") * Symbol("z_h")
        dualed[Symbol("z")] = 1 - (u_main - v_main) * Symbol("z")
        leader = Symbol("z_h")
    
    # for each original ill equation duplicate it for u and v
    for x, eq in ill_sys.items():
        dualed[uvars[x]] = eq
        dualed[vvars[x]] = eq

    # Substitute ill system variables with (u_ - v_) in each new equation
    for x in ill_sys.keys():
        for y in ill_sys.keys():
            dualed[uvars[x]] = dualed[uvars[x]].subs(y, uvars[y] - vvars[y])
            dualed[vvars[x]] = dualed[vvars[x]].subs(y, uvars[y] - vvars[y])
    
    # Expand everything
    for eq_key in list(dualed.keys()):
        dualed[eq_key] = expand(dualed[eq_key])
    
    # for each variable split the u-equation into its positive and negative terms and adjust with a correction term(I used the same type of stuff from niave)
    is_positive = lambda term: term.as_coeff_Mul()[0].is_positive
    for x in ill_sys.keys():
        args = Add.make_args(dualed[uvars[x]])
        pos_terms, neg_terms = sift(args, is_positive, binary=True)
        # u_x * v_x * (sum(positive terms) - sum(negative terms))
        correction = uvars[x] * vvars[x] * (sum(pos_terms) - sum(neg_terms))
        dualed[uvars[x]] = expand(sum(pos_terms) - correction)
        dualed[vvars[x]] = expand(-sum(neg_terms) - correction)

    ### Step 6. Sub new ill stuff into good equations:
    # in good equations replace any thing(pattern: x_...) with (u_x - v_x) that is inside the ill ones
    good_tokens = {str(k) for k in good_sys.keys()}
    transformed_good = {}
    token_pattern = re.compile(r'\bx_\w+\b')
    
    def token_replacer(match):
        token = match.group(0)
        return token if token in good_tokens else f"(u_{token} - v_{token})" #check if its good and if its not replace it
    
    for key, expr in good_sys.items():
        new_expr_str = token_pattern.sub(token_replacer, str(expr)) # replaces each one
        transformed_good[key] = expand(sympify(new_expr_str))

    ### Step 7. Combine the two systems and return:
    final_system = {}
    final_system.update(dualed)
    final_system.update(transformed_good)
    new_iv = convert_dual_rail_iv(system,final_system,iv) # For now, but later will convert initial value
    return final_system, new_iv, leader


def convert_dual_rail_iv(original_system, dual_rail_system, original_iv):
    """
    Construct initial values for a dual-railed system.

    Parameters:
        original_system (dict): {sympy.Symbol: expr} for the original system.
        dual_rail_system (dict): {sympy.Symbol: expr} for the dual-railed system.
        original_iv (dict): {sympy.Symbol: float} mapping original variables to initial values.
        
    Assumes: internal representation of dual-railed system has post vars named "u..." and negative named "v..."

    Returns:
        dict: {sympy.Symbol: float} initial values for the dual-railed system.
    """

    new_iv = {}
    original_vars = set(original_system.keys())

    for var in dual_rail_system:
        var_name = str(var)

        if var_name.startswith('u') or var_name.startswith('v'):
            # Dual-railed variable, e.g. x+, x-
            base_name = var_name[2:]
            base_symbol = Symbol(base_name)

            if base_symbol not in original_iv:
                raise ValueError(f"Missing initial value for dual-railed variable base: {base_name}")

            if var_name.startswith('u'):
                new_iv[var] = original_iv[base_symbol]
            else:
                new_iv[var] = 0.0

        elif var in original_iv:
            # Unchanged variable: copy IV directly
            new_iv[var] = original_iv[var]

        elif var_name in {"z", "z_h"}:
            # Special variables z and z_h
            new_iv[var] = 0.0

        else:
            # Fallback: treat as zero-initialized unless stated otherwise
            new_iv[var] = 0.0

    return new_iv

# Converts a passed GPAC to a CRN system
def gpac_to_crn(sys):
    # Initialize an empy list to store the total variables spotted in the system
    variable_watching = []
    # Initialize a dict to store the important information for construction the CRN
    products = {}
    rate_constants = {}
    # Delta values are stored as a dictionary of lists of dictionaries, with values being stored 
    # as [input_reactants][outputs] (so the inverse of how its typically visualized)
    delta_values = {}

    # Loop through every function in the ODE
    for x in sys.keys():

        # Append the function's associated var to a list so we can use it for building vectors later
        variable_watching.append(x)

        # Store the function as a list. If the function is just 1 CRN input, this is redundant. 
        # However, if multiple CRN inputs make up the function, we can overwrite this list with a more componentized version of that
        components = [sys[x]]

        # If multiple CRN inputs are represented in the ODE function (i.e. by them being added together),
        # seperate them out
        if sympify(sys[x]).atoms(Add):
            components = []
            for component in sympify(sys[x]).args:
                components.append(component)

        # For each seperated out set of components, we want to isolated the reactant from each other and the integer modifier
        for component in components:
            # First sympify the component for processing
            component = sympify(component)
            # Blank out the hold variable for the modifier
            hold_int = None

            # If the input is just a single reactant, no modifier, no other reactants, that must be handled
            # as a special case. We need to tell the reactant to break apart again to try to seperate out the 
            # integer modifier from being unified as a multiplication element with the reactant
            reactants = [component]
            if component.atoms(Mul):
                reactants = []
                # Look at each piece of the multiplication element and overwrite the currently being examined reactant
                # with the list of components
                for element in component.args:
                    reactants.append(element)

            true_reactants = []
            # Compile a list of things that are the actual, true variable reactants we want to focus on
            for reactant in reactants:
                # If the currently being evaluated piece is not of type symbol or power, then it must be the integer modifier, and we want to set that aside
                if not reactant.atoms(Symbol) and not reactant.atoms(Pow):
                    hold_int = reactant
                else:
                    true_reactants.append(str(reactant))
            
            # Check to see if this is already a CRN input we're tracking, or if it corresponds to a new 
            # CRN input we haven't seen yet. Then associate the corresponding output information from the 
            # ODE to the input.
            if str(true_reactants) in products.keys():
                products[str(true_reactants)].append(x)
            else:
                products[str(true_reactants)] = [x]
                # Init the delta value store as a list
                delta_values[str(true_reactants)] = []
            # Assume the hold int is the delta. 
            # Later it will be tested if some or all of the stored int is actually the rate constant
            if hold_int:
                delta_values[str(true_reactants)].append({str(x):hold_int})
            # Because the loop here doesn't touch the delta cases where the input and output cancel out for
            # a reactant (i.e. delta = 0), we can assume whenever hold_int wasn't detected at this point,
            # the proposed delata was 1
            else:                    
                delta_values[str(true_reactants)].append({str(x):1})

    # Extract the rate constant by looking for the greatest common denominator of the deltas
    for input in delta_values.keys():
        check_list = []
        for delta in delta_values[input]:
            for delta_key in delta.keys():
                check_list.append(delta[delta_key])
        rate_constant = math.gcd(*check_list)
        rate_constants[input] = rate_constant

        for delta in delta_values[input]:
            for delta_key in delta.keys():
                delta[delta_key] = int(delta[delta_key] / rate_constant)

    result_list = []

    # For all the inputs, we will now attempt to assemble a final CRN formula
    for input in products.keys():
        hold_list = []
        remaining_reactants = variable_watching.copy()

        # Add the rate constant
        hold_list.append(rate_constants[input])

        # Add the input and output
        # For every variable, if it appears in the input, write the delta's value to that slot
        input_hold_list = [0]*len(variable_watching)
        output_hold_list = [0]*len(variable_watching)
        #for variable in variable_watching:
        #    input_hold_list.append(0)
        list_input = ast.literal_eval(input)
        for reactant in list_input:
            sym_reactant = sympify(reactant)
            if sym_reactant.atoms(Pow):
                remove_reactant = str(sym_reactant.args[0])
            else:
                remove_reactant = reactant
            # Remove the reactant from the set of special cases to check if its in the input
            remaining_reactants.remove(remove_reactant)

            
            if sym_reactant.atoms(Pow):
                exp = sym_reactant.args[1]
                check_reactant = sym_reactant.args[0]
            else:
                exp = 1
                check_reactant = sym_reactant
        
            input_hold_list[variable_watching.index(str(check_reactant))] = exp

            # Assume all inputs are equivalently in the output
            output_hold_list[variable_watching.index(str(check_reactant))] = input_hold_list[variable_watching.index(str(check_reactant))]

            # Check the deltas to correct the assumption if false
            for delta_reactants in delta_values[input]:
                if str(check_reactant) in delta_reactants.keys():
                    output_hold_list[variable_watching.index(str(check_reactant))] = exp + (delta_reactants[str(check_reactant)])
            
        
        # For any reactants in the output but not the input, use delta to set their values
        for reactant in remaining_reactants:
            for delta_reactants in delta_values[input]:
                if str(reactant) in delta_reactants.keys():
                    output_hold_list[variable_watching.index(str(reactant))] = delta_reactants[str(reactant)]

        hold_list.append(input_hold_list)
        hold_list.append(output_hold_list)

        result_list.append(hold_list)

    return result_list


# Converts a passed PLPP to an ODE
def plpp_to_ode(sys):
    # Assume input is in the form [[k,[input vector],[output vector]]

    # Initialize the output for the ode with all the variables we'll need to be able to determine the function for
    ode_keys = []
    for i in range(len(sys[0][1])):
        ode_keys.append("z"+str(i))
        
    # Create a dict using all the variables so we can store the ODE functions per variable
    ode_output = dict.fromkeys(ode_keys)

    # For each reactant in the input, we're going to build out its function 
    for i in range(len(sys[0][1])):
        append_str = ""
        # For each function in the PLPP, we're going to search for occurances of the current reactant
        for plpp_reac in sys:
            # If the reactant is in the input and not the output, add the input to the function times the frequency of its occurence
            if plpp_reac[2][i] != 0 and plpp_reac[1][i] == 0:
                append_str += " + ( " + str(plpp_reac[0])
                for j in range(len(plpp_reac[1])):
                    if plpp_reac[1][j] != 0:
                        append_str += " * " + "z" + str(j) 
                append_str += " )"
            # If the reactant is in the input and not the output, subtract the input from the function times the frequency of its occurence
            if plpp_reac[1][i] != 0 and plpp_reac[2][i] == 0:
                append_str += " - ( " + str(plpp_reac[0])
                for j in range(len(plpp_reac[1])):
                    if plpp_reac[1][j] != 0:
                        append_str += " * " + "z" + str(j) 
                append_str += " )"
        # If every function in which the reactant occured, the reactant was both in the input and output (i.e. it always cancelled itself out), assign 0 as its function as nothing impacts it
        if append_str == "":
            append_str = "0"
        
        # If the first symbol in the function associated with the reactant is a plus sign, drop the plus sign as its redundant
        if len(append_str) > 1:
            if append_str[1] == "+":
                append_str = append_str[2:]

        # Write the formula to the dict of functions
        ode_output[ode_keys[i]] = append_str

    return ode_output
