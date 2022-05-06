
''' begin text
Example of a Python Jupyter worksheet.
'''  # end text

# import our packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, Markdown

''' begin text
Makes some example data.
'''  # end text

d = pd.DataFrame({
    'x': np.linspace(0, 7, num=100)
})
d['y'] = np.sin(d['x'])

'''end code'''

# display data

d

''' begin text
Plot our data.
'''  # end text

sns.lineplot(
    data=d,
    x='x',
    y='y',
)
plt.title('example plot')
plt.show()

'''end code'''

display(Markdown(f"""
Our example dataframe has {d.shape[0]} rows.
"""))

''' begin text
And that is our example.
'''  # end text

