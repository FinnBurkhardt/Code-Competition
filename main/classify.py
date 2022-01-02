from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import sys


def train(modelDir,DataDir):
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


	model = XGBClassifier(n_estimators=10,max_depth=1)
	model.fit(X_train,Y_train)

	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(Y_test,predictions)
	print("accuracy: " +str(accuracy*100.0))


	model.save_model(modelDir+".model")





	model.get_booster().dump_model(modelDir+'.txt',with_stats=True)
	with open(modelDir+'.txt','r') as f:
		txt_model =f.read()




def predict(modelDir,dataDir):

	features = np.genfromtxt(dataDir,delimiter=";")

	features = np.delete(features,(0),axis=0)
	features = np.delete(features,(0),axis=1)


	model = XGBClassifier(n_estimators=10,max_depth=1)
	model.load_model(modelDir)
	y_pred = model.predict(features)
	print(y_pred)




if __name__ == "__main__":
	args = sys.argv


	if len(args)>1:

		if args[1]=="train":
			modelDir = "xgb_model"
			dataDir = "features.csv"
			if len(args)>3:
				modelDir = args[2]
				dataDir = args[3]
			train(modelDir,dataDir)
		elif args[1]=="predict":
			modelDir = "xgb_model.model"
			dataDir = "toPredict.csv"
			if len(args)>3:
				modelDir = args[2]
				dataDir = args[3]
			
			predict(modelDir,dataDir)

