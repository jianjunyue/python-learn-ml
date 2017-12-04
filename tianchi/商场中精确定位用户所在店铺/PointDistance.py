import math

PI = 3.14159265358979323846
EARTH_RADIUS = 6378.137;# 地球半径
def rad(d):
    return d * PI / 180.0

def GetPreciseDistance(lat1,   lng1,   lat2,   lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = s*1000
    return s


GetPreciseDistance(36.088040,132.23082291,31.1187894,112.308782)