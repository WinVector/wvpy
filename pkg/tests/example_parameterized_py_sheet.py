
from wvpy.jtools import declare_task_variables


with declare_task_variables(dict()):
    x = 1

assert x == 1
del x

with declare_task_variables(globals()):
    x = 1

assert x == 7
