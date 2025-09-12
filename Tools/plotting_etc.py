import plotly.express as px
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sympy.utilities.lambdify import *
from sympy import *
from sympy import symbols, sympify
import _pickle
import datetime
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, ImplicitEuler, PIDController, Kvaerno5, Tsit5
import jax
import jax.numpy as jnp
import sympy2jax as s2j
import multiprocessing as mp
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)

'''You only have to functionize a dictionary to solve and plot it. This is not a part of the actual transformation process.
Used for solving ODE systems with scipy rather than with diffrax.'''
def functionize_dict(system):
    spfy = sympify(system)
    def sys_rhs(t, y):
        # Create a dictionary mapping variable names to their current values
        var_values = {var: value for var, value in zip(spfy.keys(), y)}

        # Evaluate each expression in the system
        dxdts = [expression.evalf(subs=var_values) for expression in spfy.values()]
        return dxdts

    # Return the system of equations and the variable names
    return sys_rhs, list(spfy.keys())


def sympy_to_diffrax_term(sympy_system):
    """
    Convert a dict of SymPy SYSTEM (var -> expression) into a diffrax.ODETerm.
    """
    symbols_list = list(sympy_system.keys())
    expressions = [sympy_system[var] for var in symbols_list]

    # Build the Sympy2JAX module
    mod = s2j.SymbolicModule(expressions)

    def vector_field(t, y, args):
        # Convert all values to jnp-compatible floats
        param_dict = {str(sym): jnp.asarray(y_i) for sym, y_i in zip(symbols_list, y)}
        return jnp.asarray(mod(**param_dict))  # Ensure result is JAX array too

    return ODETerm(vector_field)

def format_dict(dict_):
    """
    Formats a dictionary to be more readable.
    """
    formatted = ""
    for key, value in dict_.items():
        formatted += f"{key}: {value}\n"
    return formatted


# ---------- TOP-LEVEL WORKER (must not be nested) ----------
@dataclass
class _SimpleSolution:
    ts: np.ndarray
    ys: np.ndarray

# ---------- TOP-LEVEL WORKER ----------
def _solver_worker(q, solver_cls, sympy_system, ode_func, times, y0_np, time_span,
                   rtol, atol, dt0, max_steps):
    """
    Runs in a child process. Builds the term here, solves, and sends (ts, ys).
    """
    try:
        # Build the term INSIDE the child to avoid pickling closures
        if sympy_system is not None:
            term = sympy_to_diffrax_term(sympy_system)
        elif ode_func is not None:
            term = ODETerm(ode_func)  # must be a top-level callable if used
        else:
            raise ValueError("Must provide either sympy_system or ode_func.")

        # Use float64 if you enabled it globally; otherwise float32 is fine
        y0 = jnp.asarray(y0_np)  # dtype choice inherited from y0_np

        solver = solver_cls()
        kwargs = dict(
            t0=time_span[0], t1=time_span[1],
            y0=y0, dt0=dt0,
            saveat=SaveAt(ts=times),
            max_steps=max_steps
        )
        # Always set tolerances on a stepsize controller in diffrax
        kwargs["stepsize_controller"] = PIDController(rtol=rtol, atol=atol)

        soln = diffeqsolve(term, solver, **kwargs)

        # Send only pickle-friendly payload
        ts_np = np.asarray(soln.ts)
        ys_np = np.asarray(soln.ys)
        q.put(("ok", ts_np, ys_np, None))
    except Exception as e:
        # Send the error text; keep it small & pickleable
        q.put(("err", None, None, str(e)))


def _attempt_solver_with_timeout(solver_cls, times, y0_np, time_span,
                                 rtol, atol, dt0, max_steps,
                                 timeout_seconds, sympy_system, ode_func):
    """
    Run one solver attempt in a child process. Returns (simple_soln_or_None, err_or_None).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_solver_worker,
        args=(q, solver_cls, sympy_system, ode_func, times, y0_np, time_span,
              rtol, atol, dt0, max_steps)
    )
    p.start()
    try:
        # Try to get result, but only wait up to timeout_seconds
        result = q.get(timeout=timeout_seconds)
    except q.Empty:
        # Child is taking too long → kill it
        p.terminate()
        p.join()
        return None, f"timeout after {timeout_seconds}s"

    p.join()  # now safe to reap
    status, ts_np, ys_np, msg = result
    if status == "ok":
        return _SimpleSolution(ts=ts_np, ys=ys_np), None
    else:
        return None, msg
    
def jacobian_from_sympy(sympy_system, var_order):
    vars_ = list(var_order)
    exprs = [sympy_system[v] for v in vars_]
    Jsym = Matrix(exprs).jacobian(vars_)
    Jmod = s2j.SymbolicModule(list(Jsym))  # flatten rows
    n = len(vars_)
    def J(y):
        vals = {str(v): y[i] for i, v in enumerate(vars_)}
        flat = np.array(Jmod(**vals), dtype=float)
        return flat.reshape((n, n))
    return J

def solve_and_plot_ode(
    ode_func, initial_values, time_span=(0,20), num_points=101,
    showSum=False, showMax=False, var_names=None, plot=True,
    sympy_system=None, solver_timeout=20, urtol=None, uatol = None, udt0 = None
):
    """
    Try several solvers with a timeout (minutes). Return the first that succeeds.
    """
    # Time grid and initial value; pass NumPy to child (safer to pickle), convert to JAX inside worker
    times = np.linspace(time_span[0], time_span[1], num_points)
    y0_np = np.asarray(initial_values, dtype=jnp.float64)

    # J = jacobian_from_sympy(sys, sys.keys())
    # lam = np.linalg.eigvals(J(y0_np))
    # stiffness_ratio = max(lam) / min(lam)  

    # Validate inputs (but do NOT build the term here, to avoid pickling closures)
    if sympy_system is None and ode_func is None:
        raise ValueError("Must provide either a sympy_system or an ode_func.")

    # If a sympy_system is provided, ignore ode_func for the worker to avoid pickling a local closure
    ode_func_for_worker = None if sympy_system is not None else ode_func

    solvers = [Kvaerno5, ImplicitEuler, Tsit5, Dopri5]
    timeout_seconds = int(solver_timeout * 60)

    last_err = None
    for solver_cls in solvers:
        print(f'Trying solver {solver_cls}')
        soln, err = _attempt_solver_with_timeout(
            solver_cls, times, y0_np, time_span,
            rtol=urtol, atol=uatol * np.maximum(1.0, np.abs(y0_np)), dt0=udt0, max_steps=None,
            timeout_seconds=timeout_seconds,
            sympy_system=sympy_system,
            ode_func=ode_func_for_worker
        )
        if soln is not None:
            # Plotting code is unchanged
            if plot:
                _, ax = plt.subplots()
                plt.subplots_adjust(right=0.7)  # Space for checkbox widget
                lines = []

                # Plot each variable
                for i in range(soln.ys.shape[1]):
                    label = var_names[i] if var_names else f'Variable {i+1}'
                    line, = ax.plot(soln.ts, soln.ys[:, i], '-', label=label)
                    lines.append(line)

                # Plot sum of variables if requested
                if showSum:
                    total = jnp.sum(soln.ys, axis=1)
                    line, = ax.plot(soln.ts, total, '-', label="Sum")
                    lines.append(line)

                # Plot running max of total
                if showMax:
                    total = jnp.sum(soln.ys, axis=1)
                    running_max = jnp.maximum.accumulate(total)
                    line, = ax.plot(soln.ts, running_max, '-', label="Max")
                    lines.append(line)

                # Interactive checkboxes
                labels = [line.get_label() for line in lines]
                visibility = [line.get_visible() for line in lines]
                check = CheckButtons(plt.axes([0.75, 0.4, 0.2, 0.5]), labels, visibility)

                def func(label):
                    idx = labels.index(label)
                    lines[idx].set_visible(not lines[idx].get_visible())
                    plt.draw()

                check.on_clicked(func)

                # Labels and legend
                ax.set_xlabel("Time")
                ax.set_ylabel("Values")
                ax.set_title("Solution of the ODE System")
                ax.legend()
                plt.show()
            return soln
        else:
            last_err = f"{solver_cls.__name__}: {err}"

    print(f"All solvers failed. Last error: {last_err}")


'''Functionize sys, solve it using the provided initial values (iv), write all of that to a logfile.'''

def fsp(sys, iv, debug=True, log=True, time_span=(0, 20), num_points=101, showSum=False, showMax=False, logfilename='ode_log.txt', plot=True, mainvar = Symbol('x_1')):
    fsys, names = functionize_dict(sys) # functionized dict is used for solving with scipy rather than diffrax

    if log:
        write_log_before_solving(sys, iv, names, logfilename)

    # We pass both fsys and sys here, but the worker will prefer sympy_system
    # and will NOT pickle fsys (ode_func) across processes.
    dt0 = (time_span[1] - time_span[0]) / 1000.0
    soln = solve_and_plot_ode(fsys, iv, var_names=names, time_span=time_span, num_points = num_points, plot=plot, sympy_system=sys, urtol=1e-6,uatol=1e-8,udt0 = 1e-12 )
    if soln :
        leader_limit = soln.ys[-1, names.index(mainvar)] if mainvar in names else None

        if log:
            write_log_after_solving(names, soln, logfilename)

        return soln, leader_limit
    return None, None

def write_simple(sys,iv,names,filename):
    """
    Writes the given ODE system and initial values to a log file in a human-readable form before solving the ODEs.

    :param sys: The ODE system to be logged (dictionary with sympy expressions).
    :param iv: Initial values of the variables (list).
    :param names: List of variable names corresponding to the system's variables.
    :param filename: The name of the log file (default is 'ode_log.txt').
    """
    with open(filename, 'a') as log_file:
        log_file.write(f'ODE System (Before Solving), Timestamp ({datetime.datetime.now()}):\n')
        for v, eq in sys.items():
            log_file.write(f"{v}' = {eq}\n")
        log_file.write('\n\n')  # Extra space before the next section

        log_file.write("Initial Values:\n")
        for name, val in zip(names, iv):
            log_file.write(f"{name} = {val}\n")
        log_file.write('\n\n')  # Extra space before the next section

def write_log_before_solving(sys, iv, names, filename='ode_log.txt'):
    write_simple(sys,iv,names,filename)

def write_log_after_solving(names, soln, filename='ode_log.txt'):
    """
    Writes the solutions of the ODE system to a log file in a human-readable form after solving the ODEs.

    :param names: List of variable names corresponding to the system's variables.
    :param soln: Solutions of the ODE system (dictionary).
    :param filename: The name of the log file (default is 'ode_log.txt').
    """
    # Unpack the tuple returned by sample_solution into sampled_soln and selected_times
    sampled_soln, selected_times = sample_solution(soln, names)

    with open(filename, 'a') as log_file:
        log_file.write(f'Sampled Solutions (After Solving) Timestamp ({datetime.datetime.now()}):\n')
        # Log selected times first
        log_file.write(f"Selected Times: {selected_times}\n\n")

        # Now log the sampled solutions
        for var, values in sampled_soln.items():
            log_file.write(f"{var}: {values}\n")

        log_file.write('\n\n')  # Extra space before the end

# def sample_solution(soln, names):
#     """
#     Samples the solution at eight specific times over the span of the execution and returns the sampled solutions along with the selected times.
#     Two times are early, and six times are fairly late to observe stabilization.

#     :param soln: Solution object returned by solve_ivp, expected to have 't' for times and 'y' for solution values.
#     :param names: List of variable names corresponding to the solution's variables.
#     :return: A tuple containing a dictionary with sampled values for each variable at the specified times, and the list of selected times.
#     """
#     sampled_soln = {}
#     total_times = len(soln.ts)

#     # Determine sample times: 2 early, 6 late
#     early_times = [0, total_times // 8]
#     late_times = [total_times * 5 // 8, total_times * 6 // 8, total_times * 7 // 8, total_times * 15 // 16, total_times * 31 // 32, total_times - 1]
#     sampled_times = early_times + late_times

#     # Convert sampled times from indices to actual times
#     selected_times = [soln.t[t] for t in sampled_times]

#     # Sample for each variable using the provided names
#     for i, var_series in enumerate(soln.y):
#         sampled_values = [var_series[t] for t in sampled_times]
#         sampled_soln[names[i]] = sampled_values

#     return sampled_soln, selected_times

'''Above fn but diffrax'''
def sample_solution(soln, names):
    """
    Samples the solution at eight specific times over the span of the execution and returns the sampled solutions along with the selected times.
    Two times are early, and six times are fairly late to observe stabilization.

    :param soln: Diffrax solution object with 'ts' (times) and 'ys' (values with shape [n_times, n_vars]).
    :param names: List of variable names corresponding to the solution's variables.
    :return: A tuple containing a dictionary with sampled values for each variable at the specified times, and the list of selected times.
    """
    sampled_soln = {}
    total_times = len(soln.ts)

    # Indices: 2 early, 6 late
    early_indices = [0, total_times // 8]
    late_indices = [
        total_times * 5 // 8,
        total_times * 6 // 8,
        total_times * 7 // 8,
        total_times * 15 // 16,
        total_times * 31 // 32,
        total_times - 1
    ]
    sampled_indices = early_indices + late_indices

    # Convert indices to actual times
    selected_times = [soln.ts[i] for i in sampled_indices]

    # For each variable, sample values at those times
    for var_index, var_name in enumerate(names):
        sampled_values = [float(soln.ys[i][var_index]) for i in sampled_indices]
        sampled_soln[var_name] = sampled_values

    return sampled_soln, selected_times

def unpickle(filename):
    sys = {}
    try:
        with open(filename, 'rb') as file:
            sys = pickle.load(file)
        return sys
    except:
        raise Exception(f'Error during de-serialization. Are you sure file {filename} exists?')
