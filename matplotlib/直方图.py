import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

y=[21,34,5,6,78,9,5,22,34,5,6,78,9,5,22]
df = pd.DataFrame(columns=['key', 'count'])
df["count"]=y


plt.hist(df['count'], bins=4)

plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()