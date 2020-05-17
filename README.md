[wvpy](https://github.com/WinVector/wvpy) is a simple 
set of utilities for teaching data science and machine learning methods.
They are not replacements for the obvious methods in sklearn.




```python
import wvpy.util

wvpy.__version__
```




    '0.2.0'



Illustration of cross-method plan.


```python
wvpy.util.mk_cross_plan(10,2)

```




    [{'train': [1, 4, 5, 6, 8], 'test': [0, 2, 3, 7, 9]},
     {'train': [0, 2, 3, 7, 9], 'test': [1, 4, 5, 6, 8]}]




```python

```
