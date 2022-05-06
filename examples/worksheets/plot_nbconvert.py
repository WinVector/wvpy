#!/usr/bin/env python
# coding: utf-8

# Example of a Python Jupyter worksheet.

# In[ ]:


# import our packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, Markdown


# Makes some example data.

# In[ ]:


d = pd.DataFrame({
    'x': np.linspace(0, 7, num=100)
})
d['y'] = np.sin(d['x'])


# In[ ]:


# display data

d


# Plot our data.

# In[ ]:


sns.lineplot(
    data=d,
    x='x',
    y='y',
)
plt.title('example plot')
plt.show()


# In[ ]:


display(Markdown(f"""
Our example dataframe has {d.shape[0]} rows.
"""))


# And that is our example.
