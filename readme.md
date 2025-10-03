
 GPAC-TO-TPP

## This compiler transforms a bounded general-purpose analog compouter (GPAC), i.e., a polynomial initial value problem (PIVP), into a termoleclar population protocol (TPP).

## The transformation preserves the following property:

## If the input system $G$ has a variable $a(t)$ with $\lim_{t \rightarrow \infty} a(t) = \gamma$ for some real value $\gamma$ (i.e, if $G$ computes $\gamma$) then the resulting TPP, T, also has a variable $a(t)$ with the same property. That is, the transformation preserves real-number computation in the limit, as well as boundedness (inherent to TPPs).

An example of the full compilation process history can be found in Test4Out.txt: An input system computing Euler's $\gamma$ (= $.577$...) is transformed first into a chemical reaction network, and finally into a termolecular population protocol. However, it is very, very long, and the resulting population protocol is of an immense size. Future work will seek to optimize the output size.

### The transformation goes like this:

1. The input system $G$ is given along with an initial condition. The 'main variable' (which converges to some value) is supplied, and the transformation keeps track of this special variable.
2. If selected, the input is pre-processed: $G$ is transformed into $G'$ having initial value 0 for all variables, and computing the same value in the limit. (The alternative is not yet implemented - i.e. the transformations of non-zero initial values between phases. Pre-processing should always be selected, unless the input system has 0 initial values.)
3. If the resulting system is not a chemical reaction network (does not have the form $y' = p_y - q_y * y$) then a selective dual railing algorithm converts a minimal set of variables into dual-rail representation, ensuring that the reusulting system $C$ is a CRN. If pre-processing was selected, then this CRN will also have initial values uniformly 0.
4. Next, a greedy decomposition algorithm reduces the degree of $C$ to (non-homogeneously) 2, resulting in system $C_2$ also with all-0 initial values. This requires introducing a (often very large) number of auxiliary variables. There are \textit{many} variations of this decomposition strategy, and currently the strategy used here is not very 'smart'. You can find some discussion and examples [here](https://github.com/andrei-migunov/Deterministic-CRN-to-Degree-2-CRN). $C_2$ is, again, a non-homogeneously degree 2 system: some terms may have degree less than 2.
5. The degree-2 system is transformed into a homogeneously degree-3 conservative system (a termolecular population protocol).
6. If selected, the resulting systems are simulated (options: "INPUT", "CRN", "DECOMP"). 


### Sample executions:
In `compile_tests.py` you can find a few sample GPAC inputs. You can run these from, say, VSCode, by swapping out the test name in the code at the bottom of that file:
 
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()       
    #test()
    test4v2()
    #test3()

### Command line inputs to the main function (`main.compile_from_file()`) are provided as follows:

`python main.py "test_one_over_e.txt"`

You should see output indicating that for all three 'phases' of compilation, the limiting value of the main system variable is 1/e, or about .3678...
You should also see that the file test_one_over_eoutput.txt is produced, as well as test_one_over_e.pkl, a serialized version of the CompileHistory object, 
whose values can be de-seralized later on to avoid having to recompile.

See `test_one_over_e.txt` for an example of input file format and flags. 


### General warnings:

1. There are a number of unused functions hanging around from previous compilation workflows. This needs to be cleaned up, and we apologize for the inconvenience.
2. Simulation is not guaranteed. The solvers are set to time out after a few different solvers have been tried and the expected simtime (25, by default) has been reached on all of them.
Many systems may seem small at first, but are very coupled, or have very high degree terms in them that result in simply massive output systems. Whether they can be simulated successfully depends on a lot of things. Use the simulation suite at your own risk. It is suggested to take the compilation output .pkl file, unpickle it in your own code, and make your own decisions about how to simulate. 

### In future versions:

Bug fixes, and quality of life updates.
Implementation of non-zero IV compilation pathway.
Reduced system sizes.


### Some comments on installation:

You may want to do the following in a virtual environment (.venv) rather than on your main python installation. 

You may want to first install numpy 1.26.4 and lock it. Then, install everything else from the requirements.

All together,

Create and activate the NEW env:
`py -3.10 -m venv .venv`
`.\.venv\Scripts\Activate.ps1`

Be explicit about which Python/pip we’re using:
`python -m pip install -U pip setuptools wheel`

Install NumPy first to lock the ABI:
`python -m pip install --no-cache-dir "numpy==1.26.4"`

Then, install from requirements.txt:
`python -m pip install --no-cache-dir -r requirements.txt` 

Acknowledgements:


Thank you to those who have contributed or continue to contribute to this project both at the theoretical level and the software level: Nicholas Haisler (Drake), Katja Mathesius (Drake, now North Carolina State University), Garrett Provence (Drake), Khalid Mohammed (Drake), and Xiang Huang (University of Illinois - Springfield). 
