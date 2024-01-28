
from wvpy.jtools import override_task_vars


with override_task_vars(globals()):
    x = 1

assert x == 7
