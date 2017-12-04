from sklearn.ensemble import RandomForestRegressor

data=[[0,0,0],[1,1,1],[2,2,2],[1,1,1],[2,2,2],[0,0,0]]
target=[0,1,2,1,2,0]
rf = RandomForestRegressor()
rf.fit(data, target)

print(rf.predict([[1,1,1]]))
print(rf.predict([[1,1,1],[2,2,2]]))
#[ 1.]
#[ 1.  1.9]
print("-----------------")
data2=[[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
target2=[0,1,2,3,4,5]
rf2 = RandomForestRegressor()
rf2.fit(data2, target2)
print(rf2.predict([[1,1,1]]))
print(rf2.predict([[1,1,1],[2,2,2],[4,4,4]]))
#[ 0.7]
#[ 0.7  1.8  4. ]