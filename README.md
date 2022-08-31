
<a href="https://github.com/WinVector/wvpy">wvpy</a> tools for converting Jupyter notebooks to and from Python files.

Text and video tutotials here: [https://win-vector.com/2022/08/20/an-effective-personal-jupyter-data-science-workflow/](https://win-vector.com/2022/08/20/an-effective-personal-jupyter-data-science-workflow/).

Many of the data science functions have been moved to wvu [https://github.com/WinVector/wvu](https://github.com/WinVector/wvu).




<a href="https://github.com/WinVector/wvpy">wvpy</a> is a very effective personal Jupyter workflow for data science development.


<a href="https://jupyter.org">Jupyter</a> (nee IPython) workbooks are JSON documents that allow a data scientist to mix: code, markdown, results, images, and graphs. They are a great contribution to scientific reproducibility, as they can contain a number of steps that can all be re-run in batch. They serve a similar role to literate programming, SWEAVE, and rmarkdown/knitr. The main design difference is Jupyter notebooks do not separate specification from presentation, which causes a number of friction points. They are not legible without a tool (such as JupyterLab, Jupyter Notebook, Visual Studio Code, PyCharm, or other IDEs), they are fairly incompatible with source control (as they may contain images as binary blobs, and many of the tools alter the notebook on opening), and they make <code>grep</code>ing/searching difficult.

The above issues are fortunately all <em>inessential difficulties</em>. Python is a very code-oriented work environment, so most tools expose a succinct programable interface. The tooling exposed by the Python packages <a href="https://pypi.org/project/ipython/">IPython</a>, <a href="https://pypi.org/project/nbformat/">nbformat</a>, and <a href="https://pypi.org/project/nbconvert/">nbconvert</a> are very powerful and convenient. With only a little organizing code we were able to build a very powerful personal data science workflow that we have found works very well for clients.

We share this small amount of code in the package <a href="https://pypi.org/project/wvpy/">wvpy</a>. This is easiest to demonstrate in action, both in <a href="https://win-vector.com/2022/08/20/an-effective-personal-jupyter-data-science-workflow/">this article</a> and in a video demonstration <a href="https://youtu.be/cQ-tCwD4moc">here</a>.

The first feature is: converting Jupyter notebooks (which are JSON files ending with a <code>.ipynb</code> suffix) to and from simple Python code that is more compatible with source control (such as Git).

Let's start with a simple example Jupyter notebook: <a href="https://github.com/WinVector/wvpy/blob/main/examples/worksheets/plot.ipynb">plot.ipynb</a>. If we install (using a shell such as bask, or zsh) <a href="https://github.com/WinVector/wvpy">wvpy</a> <a href="https://pypi.org/project/wvpy/">from PyPi</a>.

<code>
<pre>
pip install wvpy
</pre>
</code>

And we download <a href="https://github.com/WinVector/wvpy/blob/main/examples/worksheets/plot.ipynb">plot.ipynb</a>

<code>
<pre>
wget https://raw.githubusercontent.com/WinVector/wvpy/main/examples/worksheets/plot.ipynb
</pre>
</code>

Then we can convert the Jupyter notebook to the Python formatted file as follows (we discuss this format a bit <a href="https://win-vector.com/2022/04/30/separating-code-from-presentation-in-jupyter-notebooks/">here</a>).

<code>
<pre>
python -m wvpy.pysheet --delete plot.ipynb
</pre>
</code>

The tool reports the steps it takes in the conversion.

<code>
<pre>
from "plot.ipynb" to "plot.py"
   copying previous output target "plot.py" to "plot.py~"
   converting Jupyter notebook "plot.ipynb" to Python "plot.py"
   moving input plot.ipynb to plot.ipynb~
</pre>
</code>

The resulting Python file is shown <a href="https://github.com/WinVector/wvpy/blob/main/examples/worksheets/plot.py">here</a>. The idea is: the entire file is pure Python, with the non-python blocks in multi-line strings. This file has all results and meta-data stripped out, and a small amount of whitespace regularization. This ".py" format is exactly the right format for source control, we get reliable and legible differences. In my personal practice I don't always check ".ipynb" files in to source control, but only the matching ".py" files. This discipline makes <code>grep</code>ing and searching for items in the project as easy as finding items in code.

In the ".py" file "begin text", "end text", and "end code" markers show where the Jupyter cell boundaries are. This allows reliable conversion from the ".py" file back to a Jupyter notebook. PyCharm and others have a similar notebook representation strategy.

We can convert back from ".py" to ".ipynb" as follows.

<code>
<pre>
python -m wvpy.pysheet --delete plot
</pre>
</code>

<code>
<pre>
from "plot.py" to "plot.ipynb"
   converting Python plot.py to Jupyter notebook plot.ipynb
   moving input plot.py to plot.py~
</pre>
</code>

Notice this time we did not specify the file suffix (the ".py" or ".ipynb"). The tooling checks that exactly one of these exists and converts one to another. This allows easy conversion back and forth reusing command history.

Either form of the worksheet can be executed and rendered by HTML from the command line as follows.

<code>
<pre>
python -m wvpy.render_workbook plot
</pre>
</code>

<code>
<pre>
start render_as_html "plot.ipynb"  2022-08-20 12:19:06.669369
	done render_as_html "plot.html" 2022-08-20 12:19:10.080226
</pre>
</code>

This produces what we expect to see from a Jupyter notebook as a presentation.

<img style="display:block; margin-left:auto; margin-right:auto;" src="https://win-vector.com/wp-content/uploads/2022/08/Screen-Shot-2022-08-20-at-12.45.27-PM.png" alt="Screen Shot 2022 08 20 at 12 45 27 PM" title="Screen Shot 2022-08-20 at 12.45.27 PM.png" border="0" width="338" height="565" />

There is also an option in the HTML renderer that strips out input blocks. This isn't fully presentation ready, but it makes for very good in progress work reports.

<code>
<pre>
python -m wvpy.render_workbook --strip_input plot
</pre>
</code>


<code>
<pre>
start render_as_html "plot.ipynb"  2022-08-20 12:19:35.251560
	done render_as_html "plot.html" 2022-08-20 12:19:38.478107
</pre>
</code>

This gives a simplified output as below.


<img style="display:block; margin-left:auto; margin-right:auto;" src="https://win-vector.com/wp-content/uploads/2022/08/Screen-Shot-2022-08-20-at-12.43.40-PM.png" alt="Screen Shot 2022 08 20 at 12 43 40 PM" title="Screen Shot 2022-08-20 at 12.43.40 PM.png" border="0" width="335" height="465" />

For already executed sheets one would use the standard Juypter supplied command <code>jupyter nbconvert --to html plot.ipynb</code>, the merit of the rendering here is parameterization of notebooks and stripping of input and prompt ids. The strategy here is to be lightweight stand-alone, and not a plug in such as the strategy pursued by <a href="https://github.com/mwouts/jupytext">jupytext</a> or <a href="https://www.fast.ai/2022/07/28/nbdev-v2/">nbdev</a>, or targeting fully camera ready reports via <a href="https://www.fast.ai/2022/07/28/nbdev-v2/">Quarto</a>. We feel the <a href="https://github.com/WinVector/wvpy">wvpy</a> approach maximizes productivity during development, with minimal plug-in and install burdens.

We also supply a <a href="https://github.com/WinVector/wvpy/blob/main/pkg/wvpy/jtools.py#L281">simple class for holding render tasks</a>, including inserting arbitrary initialization code for each run. This makes it very easy to render the same Jupyter workbook for different targets (say the same analysis for each city in a state) and even parallelize the rendering using standard Python tools such as <code>multiprocessing.Pool</code>. This parameterized running allows simple management of fairly large projects. If we need to run a great many variations of a notebook we use the <a href="https://github.com/WinVector/wvpy/blob/main/pkg/wvpy/jtools.py#L281">JTask container</a> and either a for loop or <code>multiprocessing.Pool</code> over the tasks in Python (remember, when we have Python we don't have to perform all steps at the GUI or even in a shell!). A small example of the method is found <a href="https://github.com/WinVector/wvpy/tree/main/examples/param_worksheet">here</a>, where a single Jupyter notebook <a href="https://github.com/WinVector/wvpy/blob/main/examples/param_worksheet/ParamExample.ipynb">ParamExample.ipynb</a> is used by <a href="https://github.com/WinVector/wvpy/blob/main/examples/param_worksheet/run_examples.py">run_examples.py</a> to produce the multiple per-date HTML, PDF, and PNG files found in the <a href="https://github.com/WinVector/wvpy/tree/main/examples/param_worksheet">directory</a>.

We have found the quickest development workflow is to work with the ".ipynb" Jupyter notebooks (usually in Visual Studio Code, and settng any values that were supposed to come from the <code>wvpy.render_workbook</code> by hand after checking they are not set in <code>globals()</code>). Then when the worksheet is working we convert it to ".py" using <code>wvpy.pysheet</code> and check that in to source control. 

As a side-note, I find Python is a developer first community, which is very refreshing. Capabilities (such as Jupyter, nbconvert, and nbformat) are released as code under generous open source licenses and documentation instead of being trapped in monolithic applications. This means one can take advantage of their capabilities using only a small amount of code. And under the mentioned assumption that Python is a developer first community, small amounts of code are considered easy integrations. wvpy is offered in the same spirit, it is available for use from PyPi <a href="https://pypi.org/project/wvpy/">here</a> under a BSD 3-clause License and has it code available here for re-use or adaption <a href="https://github.com/WinVector/wvpy">here</a> under the same license. It isn't a big project, but it has made working on client projects and teaching data science a bit easier for me.

<a href="https://win-vector.com">Win Vector LLC</a> will be offering private (and hopefully someday public) training on the work flow (including notebook parameterization to run many jobs from a single source, use of <code>multiprocessing.Pool</code> for speedup, and <code>IPython.display.display; IPython.display.Markdown</code> for custom results) going forward.
