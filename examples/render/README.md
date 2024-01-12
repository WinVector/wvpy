
```
python -m wvpy.render_workbook plt_example.py
```

```
python -m wvpy.render_workbook --pytxt plt_example.py
```

```
python -m wvpy.render_workbook --pytxt plt_example_batch.py
```

```
python -m wvpy.render_workbook --pytxt plt_example.py --init "
import matplotlib
import warnings
matplotlib.use('Agg')
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*FigureCanvasAgg.*',
    )
"
```

