
from wvpy.jtools import task_vars


with task_vars(globals()):
    x = 1

assert x == 7
