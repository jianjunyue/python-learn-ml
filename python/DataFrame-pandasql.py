import pandas as pd
import numpy as np
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())
data_df = pd.DataFrame({'row_id': [1, 2, 3, 4, 5],
                   'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})

sql="select * from data_df where total_bill>20"
table=pysqldf(sql)
print(table.head())