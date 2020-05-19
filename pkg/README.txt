Win Vector LLC extras for teaching data science in Python 3

pip install wvpy
# or: pip install https://github.com/WinVector/wvpy/raw/master/dist/wvpy-0.2.1.tar.gz

~/anaconda3/bin/python3

import wvpy.util
wvpy.util.mk_cross_plan(10,2)

[{'train': [0, 2, 3, 4, 5], 'test': [1, 6, 7, 8, 9]},
 {'train': [1, 6, 7, 8, 9], 'test': [0, 2, 3, 4, 5]}]


