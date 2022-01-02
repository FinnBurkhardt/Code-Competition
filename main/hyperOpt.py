from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm



features = np.genfromtxt("features.csv",delimiter=";")

features = np.delete(features,(0),axis=0)
features = np.delete(features,(0),axis=1)

print(np.unique(features.T[-1],return_counts=True))

np.random.shuffle(features)


testSize = int(features.shape[0]*0.2)
trainSize = int(features.shape[0]*0.8)

X_train = features[:-testSize,]
X_test = features[trainSize:,]
Y_train = features[:-testSize,-1]
Y_test = features[trainSize:,-1]

X_train = np.delete(X_train,(-1),axis=1)
X_test = np.delete(X_test,(-1),axis=1)


es_=[]
ds_=[]
as_=[]

for e in tqdm(range(100)):
	for d in range(15):

		model = XGBClassifier(n_estimators=e,max_depth=d)
		model.fit(X_train,Y_train)

		y_pred = model.predict(X_test)
		predictions = [round(value) for value in y_pred]

		accuracy = accuracy_score(Y_test,predictions)
		es_.append(e)
		ds_.append(d)
		as_.append(accuracy)

		print("accuracy: " +str(accuracy*100.0))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(es_,ds_,as_)

plt.show()

#model.get_booster().dump_model('xgb_model.txt',with_stats=True)
#with open('xgb_model.txt','r') as f:
#	txt_model =f.read()
#print(txt_model)



