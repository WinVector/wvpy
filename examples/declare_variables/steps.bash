
rm -f *.txt
rm -f *.html
rm -f *~

python -m wvpy.render_workbook initial_example.ipynb
python -m wvpy.render_workbook --strip_input initial_example.ipynb

rm -f *.html
python -m wvpy.pysheet --delete initial_example 
python -m wvpy.render_workbook initial_example
python -m wvpy.render_workbook --pytxt initial_example
python -m wvpy.pysheet --delete initial_example

rm -f *.txt
rm -f *.html

python cities_example_01.py



rm -f *.html
python cities_example_02.py



rm -f *.html
python -m wvpy.pysheet declare_example.ipynb
python cities_example_03.py



rm -f *.txt
rm -f *.html
rm -f *~
