
This directory is a sequence of example of using the wvpy package to render and parameterize Jupyter worksheets.


The first sheet can be rendered with:

```
python -m wvpy.render weather_example_0.ipynb
```

This processes `weather_example_0.ipynb` to produce `weather_example_0.html` and a result `.csv` file.




The next can be run with:

```
python run_weather_reports_1.py
```


This processes `weather_example_1.ipynb` to produce one HTML and CSV file for each of a number of states. What we are doing is running a for loop over a Jupyter notebook.


The next can be run with:

```
python run_weather_reports_2.py
```


This processes `weather_example_2.ipynb` to produce one HTML and CSV file for each of a number of states. What we are doing is running a for loop over a Jupyter notebook.


The final example shows the same multiple location process, with the Jupyter dependency removed (if that is what your production/ops partners want).


```
python -m wvpy.pysheet weather_example_3.ipynb
python run_weather_reports_3.py
```

(Note: the Jupyter to Python conversion can be reversed with: `python -m wvpy.pysheet weather_example_3.py`, please see here for more details: https://win-vector.com/2022/08/20/an-effective-personal-jupyter-data-science-workflow/ , https://youtu.be/cQ-tCwD4moc?si=wEbxtfHQErH-V3mc .)

A short video on the method can be found here: https://www.youtube.com/watch?v=uU3ELUCoLaw , and a blog post explaining the method can be found here: https://win-vector.com/2024/02/06/use-jupyter-notebooks-inside-for-loops/ . A longer video on the method can be found here: https://www.youtube.com/watch?v=JHCeHUn5bmw .


