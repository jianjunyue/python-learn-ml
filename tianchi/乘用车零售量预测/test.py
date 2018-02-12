#201201开始，计算月差 201304
def getMonth(sale_date):
    sale_date=sale_date-201201
    year=sale_date/100
    count=int(year)*12
    month=year-int(year)
    count =count+month*100
    return int(count)

print(getMonth(201212))
