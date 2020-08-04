[wvpy](https://github.com/WinVector/wvpy) is a simple 
set of utilities for teaching data science and machine learning methods.
They are not replacements for the obvious methods in sklearn.




```python
import wvpy.util

wvpy.__version__
```




    '0.2.2'



Illustration of cross-method plan.


```python
wvpy.util.mk_cross_plan(10,2)




```




    [{'train': [0, 1, 2, 4, 7], 'test': [3, 5, 6, 8, 9]},
     {'train': [3, 5, 6, 8, 9], 'test': [0, 1, 2, 4, 7]}]



Plotting example


```python
import pandas
```


```python
d = pandas.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [False, False, True, True, False]
})
```


```python
help(wvpy.util.plot_roc)
```

    Help on function plot_roc in module wvpy.util:
    
    plot_roc(prediction, istrue, title='Receiver operating characteristic plot', *, truth_target=True)
        Plot a ROC curve of numeric prediction against boolean istrue.
        
        :param prediction: column of numeric predictions
        :param istrue: column of items to predict
        :param title: plot title
        :param truth_target: value to condider target or true.
        :return: calculated area under the curve, plot produced by call.
        
        Example:
        
        import pandas
        import wvpy.util
        
        d = pandas.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [False, False, True, True, False]
        })
        
        wvpy.util.plot_roc(
            prediction=d['x'],
            istrue=d['y'],
        )
    



```python
wvpy.util.plot_roc(
    prediction=d.x,
    istrue=d.y,
    title='Example ROC plot')
```


![png](output_8_0.png)





    0.6666666666666667




```python
help(wvpy.util.threshold_plot)
```

    Help on function threshold_plot in module wvpy.util:
    
    threshold_plot(d: pandas.core.frame.DataFrame, pred_var, truth_var, truth_target=True, threshold_range=(-inf, inf), plotvars=('precision', 'recall'), title='Measures as a function of threshold')
        Produce multiple facet plot relating the performance of using a threshold greater than or equal to
        different values at predicting a truth target.
        
        :param d: pandas.DataFrame to plot
        :param pred_var: name of column of numeric predictions
        :param truth_var: name of column with reference truth
        :param truth_target: value considered true
        :param threshold_range: x-axis range to plot
        :param plotvars: list of metrics to plot, must come from ['threshold', 'count', 'fraction', 'precision',
            'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate',
            'enrichment', 'gain', 'lift', 'recall', 'sensitivity', 'specificity']
        :param title: title for plot
        :return: None, plot produced as a side effect
        
        Example:
        
        import pandas
        import wvpy.util
        
        d = pandas.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [False, False, True, True, False]
        })
        
        wvpy.util.threshold_plot(
            d,
            pred_var='x',
            truth_var='y',
            plotvars=("sensitivity", "specificity", "fraction"),
        )
    



```python
wvpy.util.threshold_plot(
        d,
        pred_var='x',
        truth_var='y',
        plotvars=("sensitivity", "specificity", "fraction"),
    )
```


![png](output_10_0.png)



```python

```
