# General imports
import ctypes
import os
# Project specific imports

# Imports from internal libraries


print(os.path.dirname(os.path.abspath(__file__)))

add_lib = ctypes.CDLL("/home/erikj/projects/insidrug/py_proj/erikj/temp_and_demos/libadd1.so")
add_one = add_lib.add_one

add_one.argtypes = [ctypes.POINTER(ctypes.c_int)]

x = ctypes.c_int(10)
print(x)
# add_one(ctypes.byref(x))
add_one(x)
print(x)