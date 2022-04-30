#!/usr/bin/env python
# coding: utf-8

# Example of a Python Jupyter worksheet.

# In[1]:


# import our packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Makes some example data.

# In[2]:


d = pd.DataFrame({
    'x': np.linspace(0, 7, num=100)
})
d['y'] = np.sin(d['x'])


# In[3]:


# display data

d


# Plot our data.

# In[4]:


sns.lineplot(
    data=d,
    x='x',
    y='y',
)
plt.title('example plot')
plt.show()


# And that is our example.
