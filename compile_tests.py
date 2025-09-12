from main import *


'''Compiles the example from Haisler et. al 2025.'''
def test():
    input = sympify({"a":"b-a*b-d",
              "b": "c + a-a*b**2",
              "c": "b*c - b*c**2+.1*d",
              "d":"c-c*d"})
    
    iv = sympify({"a":3,"b":2,"c":3,"d":1})
    ch = compile(input,Symbol("a"),iv,pre_process=True, cache_filename = "Test1CompileHistory.pkl", filename="Test1Out.txt", pre_process = True) # Input system, Primary variable, Input IV

def test2():
    return compile_from_file("test_compile_in.txt")

'''Compiles a system that computes (var a) 1/e.'''
def test3():
    input = sympify({"a":"1-a",
            "b": "1-b-a+a*b",
            "c": "1-c+b*c",
            "d":"1-c*d"})
    
    iv = sympify({"a":0,"b":0,"c":0,"d":0})
    ch = compile(input,Symbol("d"),iv, cache_filename = "Test3CompileHistory.pkl", filename="Test3Out.txt") # Input system, Primary variable, Input IV

'''Compiles a system that computes (var a) Euler's Gamma via variable gam.'''
def test4():

    
    input = sympify({"f":"w",
                 "g":"p*q*v",
                 "w":"-w + u + v",
                 "u":"-u + r*v",
                 "v":"-v",
                 "r":"-r**2",
                 "p":"p*v",
                 "q":"v-q",
                 "e":"1-e",
                 "E":"(1-E)*(1-e)",
                 "e_1":"1-(1-E)*e_1",
                 "e_n":"1-(e_1*e_n)",
                 "ginv":"1-(1-e_n*(f-g))*ginv",
                 "gam":"1-gam*ginv"})
    iv = sympify({"f":0,"g":0,"w":0,"u":0,"v":1,"r":1,"p":1,"q":0,"e":0,"E":0,"e_1":0,"e_n":0,"ginv":0,"gam":0})

    ch = compile(input,Symbol("gam"),iv, cache_filename = "Test4CompileHistory.pkl", filename="Test4Out.txt",pre_process= True) # Input system, Primary variable, Input IV


'''Compiles a system that computes Omega = 0.567143290409783... via variable d.'''
def test5():
    input = sympify({
"a":"(1-c)*(1-b)*(1-d)",
"b":"(1-b)**2 * (1-c)*(1-d)",
"c": "(1-c)**3 * (1-b)* (1-d)",
"d":"1-d"
})
    
    iv = sympify({"a":0,"b":0,"c":0,"d":0})
    ch = compile(input,Symbol("a"),iv, cache_filename = "Test5CompileHistory.pkl", filename="Test5OmegaOut.txt") # Input system, Primary variable, Input IV

    #fsp(ch.crn,list(ch.crn_iv.values()),time_span=(0,100))
    x= "breakpoint"

'''Test command line with .txt inputs version.'''
def test6():
    compile_from_file("test_one_over_e.txt")



'''Compiles a system that computes (var a) Euler's Gamma.'''
def test4v2():
    input = sympify({"f":"w",
                 "g":"p*q*v",
                 "w":"-w + u + v",
                 "u":"-u + r*v",
                 "v":"-v",
                 "r":"-r**2",
                 "p":"p*v",
                 "q":"v-q",
                 "e":"1-e",
                 "E":"(1-E)*(1-e)",
                 "e_1":"1-(1-E)*e_1",
                 "e_n":"1-(e_1*e_n)",
                 "ginv":"1-(1-e_n*(f-g))*ginv",
                 "gam":"1-gam*ginv"})
    iv = sympify({"f":0,"g":0,"w":0,"u":0,"v":1,"r":1,"p":1,"q":0,"e":0,"E":0,"e_1":0,"e_n":0,"ginv":0,"gam":0})

    ch = compile(input,Symbol("gam"),iv, pre_process=True, cache_filename = "Test4CompileHistory.pkl", filename="Test4Out.txt") # Input system, Primary variable, Input IV



if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()       
    #test()
    test4v2()
    #test3()