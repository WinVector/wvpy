
# clean up any remaining derived files
rm -f *.txt
rm -f *.html
rm -f *~


# show rendering of workbook to HTML
python -m wvpy.render_workbook initial_example.ipynb
open initial_example.html 


# show rendering of workbook to HTML with input cell stripping
python -m wvpy.render_workbook --strip_input initial_example.ipynb
open initial_example.html 



# clean up
rm -f *.html


# convert Jupyter/IPython sheet to simple Python .py
python -m wvpy.pysheet --delete initial_example
cat initial_example.py



# render .py file (has implicit converstion back to Jupyter!)
python -m wvpy.render_workbook initial_example
open initial_example.html

# render with -pytxt option to prevent use of Jupter
python -m wvpy.render_workbook --pytxt initial_example
open initial_example.txt

# convert .py back to Jupyter notebook
python -m wvpy.pysheet --delete initial_example




# clean up
rm -f *.txt
rm -f *.html
rm -f *~


# render notebook with different cities substituted in
python cities_example_01.py
# show results
open *.html





# clean up
rm -f *.html

# render notebook in parallel for different cities
python cities_example_02.py
# show results
open *.html




# clean up
rm -f *.html


# convert to .py without deleting original
python -m wvpy.pysheet declare_example.ipynb
# render notebook in parallel without Jupyter dependenceis
python cities_example_03.py
# show results
open *.txt



# clean up
rm -f *.txt
rm -f *.html
rm -f *~
rm declare_example.py

