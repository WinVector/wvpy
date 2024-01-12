import matplotlib
import warnings
matplotlib.use('Agg')
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*FigureCanvasAgg.*',
    )

import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()

print(1 + 1)

