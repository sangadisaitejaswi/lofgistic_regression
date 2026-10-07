import pandas as pd
data=pd.read_csv(r"C:\Users\HARIKA\Downloads\iris.csv")

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
print(x,type(x))

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('Accuracy',accuracy_score(ytest,ypred)*100)
print(precision_score(ytest,ypred,average='micro')*100)
print(recall_score(ytest,ypred,average='micro')*100)

print(model.predict([[1.2,2.3,4.2,0.8]]))
