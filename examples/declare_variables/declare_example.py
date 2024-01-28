
""" begin text
Example parameterized Jupyter notebook.
"""  # end text

from wvpy.jtools import declare_task_variables


"""end code"""

with declare_task_variables(globals()):
    city = "Los Angeles"


"""end code"""

print(f"Analysis and results for {city}.")

