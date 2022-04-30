
''' begin text
Example of a Python Jupyter worksheet.
'''  # end text

# import our packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

''' begin text
Makes some example data.
'''  # end text

d = pd.DataFrame({
    'x': np.linspace(0, 7, num=100)
})
d['y'] = np.sin(d['x'])

'''end code'''

# disply data

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

''' begin text
And that is our example.
'''  # end text

