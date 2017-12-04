import datetime

dtstr = '2014-02-14'
d= datetime.strptime(dtstr, "%Y-%m-%d").date().weekday()
print(d)