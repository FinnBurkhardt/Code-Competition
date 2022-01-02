import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase, ascii_uppercase
from scipy.stats import binned_statistic, chi2_contingency, chi2, f_oneway,pearsonr
from sklearn.impute import KNNImputer
from scipy.ndimage.filters import uniform_filter1d
import sys
import warnings
warnings.filterwarnings('ignore')










def getAverageFuelConsumption(df):
	#prints average value of fuel_consumption

	
	if(df['fuel_consumption'].dtype=='object'):

		is_statusRetired = df['status'] == 'retired'
		is_statusDeclined = df['status'] == 'declined'
		is_statusWaiting = df['status'] == 'waiting'

		toBeZero = ((is_statusWaiting) | (is_statusDeclined) | (is_statusRetired))
		toBeZero = toBeZero[toBeZero==True].index

		substring_list = []




		for s in ascii_uppercase+ascii_lowercase: #Set all Values that are not pure numbers to 'None'
			
			
			df.loc[df['fuel_consumption'].str.contains(s),'fuel_consumption']='None'





		df['fuel_consumption']= pd.to_numeric(df['fuel_consumption'],errors='coerce')
		print(df.loc[df['status']!='finished']['fuel_consumption'].mean())
		
		



def correctDate(df):
	#replaces invalid values of 'race_driven'

	df.loc[(df['race_driven']=='0000-00-00 00:00:00'),'race_driven']='01.01.2000 00:00'
	



def setFuelConsumptionToZeroIfNotFinished(df):
	#Sets the fuel_consumption to zero if the race is not finished

	df.loc[(df['status']=='declined') & (df['fuel_consumption']!=0),'fuel_consumption']=0
	df.loc[(df['status']=='retired') & (df['fuel_consumption']!=0),'fuel_consumption']=0

def makeBrokenToNone(df):
	#sets invalid values for fuel_consumption to 'None'

	for s in ascii_uppercase+ascii_lowercase: #Set all Values that are not pure numbers to 'None'
		df.loc[df['fuel_consumption'].str.contains(s),'fuel_consumption']='None'

def makeBrokenToNan(df):
	#sets invalid values for fuel_consumption to Nan

	for s in ascii_uppercase+ascii_lowercase: #Set all Values that are not pure numbers to 'None'
		df.loc[df['fuel_consumption'].str.contains(s),'fuel_consumption']='None'


	df.loc[df['fuel_consumption']=='None','fuel_consumption']=np.nan







def detectMissingData(df):
	#set fuel_consumption to 0 if value if invalid and to 1 if it is valid

	for s in ascii_uppercase+ascii_lowercase: #Set all Values that are not pure numbers to 'None'
		df.loc[df['fuel_consumption'].str.contains(s),'fuel_consumption']='None'
	df.loc[df['fuel_consumption']!='None','fuel_consumption']=1
	df.loc[df['fuel_consumption']=='None','fuel_consumption']=0
	return df



def runChi2Test(df,A,sig):
	#run Chi2 Test for column A and fuel_consumption on significance level sig

	contigency = pd.crosstab(df[A].to_numpy(),df['fuel_consumption'].to_numpy())
	chi,pval,dof,expected = chi2_contingency(contigency)
	prob = 1-sig
	critical_value = chi2.ppf(prob,dof)
	if chi > critical_value:
		return 1
	else:
		return 0


def anova(df,A,sig):
	#runs anova for column A and fuel_consumption on significance level sig

	f_value,pvalue = f_oneway(df[A],df['fuel_consumption'])
	if(pvalue<sig):
		return 1
	else:
		return 0



def pearson(df,A,sig):
	#runs pearson correlation test for column A and fuel_consumption on significance level sig

	df[A] = df[A].astype('float64')
	df['fuel_consumption'] = df['fuel_consumption'].astype('float64')
	df.replace([np.inf,-np.inf],np.nan,inplace=True)
	df.dropna()
	r,pvalue = pearsonr(df[A],df['fuel_consumption'])
	if(pvalue<sig):
		return 1
	else:
		return 0






def customKNNMetric(X,Y,**kwds):


	counter = 7
	for i in range(len(X)):
		if(X[i]==Y[i]):
			counter = counter-1
		elif(i==3):
			counter = counter-(1-((X[i]-Y[i])**2)/1000000)
	return counter




def imputeKNN(df):
	#Imputes invalid values for fuel_consumption

	print("Start Imputaton")
	imputer = KNNImputer(n_neighbors=3,missing_values=np.nan,metric=customKNNMetric)

	df.replace("",np.nan ,inplace=True)
	df.dropna(subset=["race_driven"],inplace=True)

	df = df.reset_index(drop=True)


	dfTemp = df.copy()


	dfTemp = dfTemp.drop('race_driven',axis=1)
	dfTemp = dfTemp.drop('race_created',axis=1)
	dfTemp = dfTemp.drop('forecast',axis=1)
	dfTemp = dfTemp.drop('status',axis=1)

	dfTemp['weather'] =  dfTemp['weather'].astype('category')
	dfTemp['weather'] =  dfTemp['weather'].cat.codes
	dfTemp.loc[dfTemp['fuel_consumption']=='None','fuel_consumption']=np.nan
	imputed = imputer.fit_transform(dfTemp)
	df_imputed = pd.DataFrame(imputed,columns = dfTemp.columns)
	df_imputed['race_driven']=df['race_driven']
	df_imputed['race_created']=df['race_created']
	df_imputed['forecast']=df['forecast']
	print("Imputaton finished")
	df_imputed.to_csv("imputed_data.csv",sep=";")
	return df_imputed









def testMARvsMCAR(df):
	#tests if missing data mechanism is MAR or MCAR

	sig =0.05
	makeBrokenToNone(df)
	detectMissingData(df)
	if(pearson(df.copy(),'money',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: Money und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: Money und MissingFuelConsumption sind unabhängig")
	if(runChi2Test(df.copy(),'track_id',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: track_id und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: track_id und MissingFuelConsumption sind unabhängig")

	if(runChi2Test(df.copy(),'challenger',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: challenger und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: challenger und MissingFuelConsumption sind unabhängig")

	if(runChi2Test(df.copy(),'opponent',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: opponent und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: opponent und MissingFuelConsumption sind unabhängig")
	if(runChi2Test(df.copy(),'winner',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: winner und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: winner und MissingFuelConsumption sind unabhängig")
	if(runChi2Test(df.copy(),'status',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: status und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: status und MissingFuelConsumption sind unabhängig")
	if(runChi2Test(df.copy(),'weather',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: weather und MissingFuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: weather und MissingFuelConsumption sind unabhängig")



def testFuelConsumptionCorrelation(df):
	#tests correlation between fuel_consumption and the other columns

	sig=0.05
	makeBrokenToNan(df)
	df=df.dropna()
	
	df['weather'] = df['weather'].astype('category')	
	df['weather'] = df['weather'].cat.codes
	if(anova(df.copy(),'track_id',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: track_id und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: track_id und FuelConsumption sind unabhängig")
	if(anova(df.copy(),'challenger',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: challenger und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: challenger und FuelConsumption sind unabhängig")
	if(anova(df.copy(),'opponent',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: opponent und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: opponent und FuelConsumption sind unabhängig")
	if(anova(df.copy(),'winner',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: winner und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: winner und FuelConsumption sind unabhängig")
	if(anova(df.copy(),'weather',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: weather und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: weather und FuelConsumption sind unabhängig")
	if(pearson(df.copy(),'money',sig)):
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: money und FuelConsumption sind abhängig")
	else:
		print("Auf dem Signifikazniveau "+str(sig)+ " wird angenommen: money und FuelConsumption sind unabhängig")


	
def plotTopTrack(df, N, makeStationary=True):
	#plots track popularity


	df['race_driven']=pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']=df['race_driven'].dt.date

	df = df[df['status']=='finished']
	

	dates = df['race_driven'].unique()
	dates = np.sort(dates)


	data_tracks = np.array([dates,np.zeros((15,dates.shape[0]))])

	for counter,row in df.iterrows():
		i=np.where(data_tracks[0]==row['race_driven'])
		data_tracks[1][int(row['track_id'])-1][i] +=1

	

	track1=[]
	track2=[]
	track3=[]
	track4=[]
	track5=[]
	track6=[]
	track7=[]
	track8=[]
	track9=[]
	track10=[]
	track11=[]
	track12=[]
	track13=[]
	track14=[]

	for i in range(len(data_tracks[1][1])):
		n=1
		if(makeStationary):
			n=0
			for t in range(14):
				n+=data_tracks[1][t][i]
			

		
		track14.append(data_tracks[1][13][i]/n)
		track13.append(data_tracks[1][12][i]/n)
		track12.append(data_tracks[1][11][i]/n)
		track11.append(data_tracks[1][10][i]/n)
		track10.append(data_tracks[1][9][i]/n)
		track9.append(data_tracks[1][8][i]/n)
		track8.append(data_tracks[1][7][i]/n)
		track7.append(data_tracks[1][6][i]/n)
		track6.append(data_tracks[1][5][i]/n)
		track5.append(data_tracks[1][4][i]/n)
		track4.append(data_tracks[1][3][i]/n)
		track3.append(data_tracks[1][2][i]/n)
		track2.append(data_tracks[1][1][i]/n)
		track1.append(data_tracks[1][0][i]/n)

	x = np.linspace(0, len(track1)-1,len(track1) )

	
	


	track1 = uniform_filter1d(np.array(track1),size=N)
	track2 = uniform_filter1d(np.array(track2),size=N)
	track3 = uniform_filter1d(np.array(track3),size=N)
	track4 = uniform_filter1d(np.array(track4),size=N)
	track5 = uniform_filter1d(np.array(track5),size=N)
	track6 = uniform_filter1d(np.array(track6),size=N)
	track7 = uniform_filter1d(np.array(track7),size=N)
	track8 = uniform_filter1d(np.array(track8),size=N)
	track9 = uniform_filter1d(np.array(track9),size=N)
	track10 = uniform_filter1d(np.array(track10),size=N)
	track11 = uniform_filter1d(np.array(track11),size=N)
	track12 = uniform_filter1d(np.array(track12),size=N)
	track13 = uniform_filter1d(np.array(track13),size=N)
	track14 = uniform_filter1d(np.array(track14),size=N)
	


	

	
	plt.plot(dates,track3,'y-',label="track3")
	plt.plot(dates,track4,label="track4")
	plt.plot(dates,track5,label="track5")
	plt.plot(dates,track6,label="track6")
	plt.plot(dates,track7,label="track7")
	plt.plot(dates,track8,label="track8")
	plt.plot(dates,track9,label="track9")
	plt.plot(dates,track10,label="track10")
	plt.plot(dates,track11,label="track11")
	plt.plot(dates,track12,'b-',label="track12")
	plt.plot(dates,track13,label="track13")
	plt.plot(dates,track14,label="track14")
	#plt.legend()
	if(makeStationary):
		plt.yscale('linear')
		plt.xlabel('Time')
		plt.ylabel('Portion of Races')
	else:
		plt.yscale('log')
		plt.xlabel('Time')
		plt.ylabel('Races per day (log scale)')


	plt.show()


def plotRaces(df,N,useLogScale=False):
	#plots Number of races per day

	df = df[df['status']=='finished']
	

	earliest = min(df['race_driven'])
	latest = max(df['race_driven'])

	times = pd.date_range(earliest,latest,freq='D')


	data_races = [times,[0]*(times.size)]


	for counter,row in df.iterrows():
		i = data_races[0].get_loc(row['race_driven'].date().strftime('%Y-%m-%d'))
		data_races[1][i] = data_races[1][i]+1



	x = np.linspace(0, len(data_races[0])-1,len(data_races[0]) )

	races = uniform_filter1d(np.array(data_races[1]),size=N)

	plt.plot(np.array(times),races)
	if useLogScale:
		plt.ylabel('Number of races per day (log)')
		plt.yscale('log')
	else:
		plt.ylabel('Number of races per day (linear)')
		plt.yscale('linear')
	plt.xlabel('Time')
	plt.show()



def getStats():
	#calculates and saves statistics in preperation for feature engeneering

	df = pd.read_csv('imputed_data.csv',sep=";",header=0)

	players = []



	for _,row in df.iterrows():
		if( (not row['challenger'] in players) and (row['challenger']!=0)):
			players.append(int(row['challenger']))



		if(not row['opponent'] in players and row['opponent']!=0):
			players.append(int(row['opponent']))

	df['race_driven']=pd.to_datetime(df['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']= df['race_driven'].dt.date

	dates = df['race_driven'].unique()


	playerId_wins = np.zeros((len(players),len(dates)))
	playerId_winrate = np.zeros((len(players),len(dates)))
	playerId_winloserate = np.zeros((len(players),len(dates)))
	playerId_trackCounts = np.zeros((len(players),len(dates),15))
	playerId_avgFuelCosumption = np.zeros((len(players),len(dates),15))

	

	df['track_id'] = df['track_id'].astype('int32')
	df['fuel_consumption'] = df['fuel_consumption'].astype('float32')



	for k,date in enumerate(dates):
		dfDate = df[df['race_driven']==date]
		for i,p in enumerate(players):
			win = 0
			races = 0


			for _,row in dfDate.iterrows():

				if ((p == row['challenger']) or (p==row['opponent'])):
					races=races+1

					
					playerId_trackCounts[i,k,row['track_id']] += 1
					playerId_avgFuelCosumption[i,k,row['track_id']] += 1


					if p==row['winner']:
						win=win+1



			
			if(not races==0):
				
				playerId_wins[i,k] = win
				playerId_winrate[i,k] = (win/races)
				playerId_winloserate[i,k] = ((2*win-races)/races)
				
		
	
		

				
				
	






	np.savetxt("playerId.csv",players,delimiter=";")
	np.savetxt("playerId_wins.csv",playerId_wins,delimiter=";")
	np.savetxt("playerId_winrate.csv",playerId_winrate,delimiter=";")
	np.savetxt("playerId_winloserate.csv",playerId_winloserate,delimiter=";")
	np.savetxt("playerId_trackCounts.csv",playerId_trackCounts.reshape(playerId_trackCounts.shape[0],-1),delimiter=";")
	np.savetxt("playerId_avgFuelCosumption.csv",playerId_avgFuelCosumption.reshape(playerId_avgFuelCosumption.shape[0],-1),delimiter=";")






def createFeatures():
	#creates features for training a classifier

	
	df = pd.read_csv('imputed_data.csv',sep=";",header=0)
	players = np.genfromtxt('playerId.csv',delimiter=";").astype('int32').astype('str')
	playerId_wins = np.genfromtxt('playerId_wins.csv',delimiter=";")
	playerId_winrate = np.genfromtxt('playerId_winrate.csv',delimiter=";")
	playerId_trackCountsArray = np.genfromtxt("playerId_trackCounts.csv",delimiter=";")
	playerId_avgFuelCosumptionArray = np.genfromtxt("playerId_avgFuelCosumption.csv",delimiter=";")


	playerId_trackCountsArray = playerId_trackCountsArray.reshape(playerId_trackCountsArray.shape[0],playerId_trackCountsArray.shape[1]//15,15)
	playerId_avgFuelCosumptionArray = playerId_avgFuelCosumptionArray.reshape(playerId_avgFuelCosumptionArray.shape[0],playerId_avgFuelCosumptionArray.shape[1]//15,15)

	
	df['race_driven']=pd.to_datetime(df['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']= df['race_driven'].dt.date


	

	dates = df['race_driven'].unique()
	playerId_wins = pd.DataFrame(playerId_wins.T, columns=players)
	playerId_winrate = pd.DataFrame(playerId_winrate.T, columns=players)
	playerId_trackCounts= pd.DataFrame(np.zeros((playerId_trackCountsArray.shape[1],playerId_trackCountsArray.shape[0])), columns=players)
	playerId_avgFuelCosumption= pd.DataFrame(np.zeros((playerId_avgFuelCosumptionArray.shape[1],playerId_avgFuelCosumptionArray.shape[0])), columns=players)

	for i in range(playerId_trackCountsArray.shape[1]):
		for p in players:
			playerIndex = np.where(players==p)[0][0]
			playerId_trackCounts.iat[i,playerIndex] = np.array(playerId_trackCountsArray[playerIndex][i],dtype=object)

	for i in range(playerId_avgFuelCosumptionArray.shape[1]):
		for p in players:
			playerIndex = np.where(players==p)[0][0]
			playerId_avgFuelCosumption.iat[i,playerIndex] = np.array(playerId_avgFuelCosumptionArray[playerIndex][i],dtype=object)




	dates = df['race_driven'].unique()



	playerId_wins.set_index(dates)

	

	featureDataFrame = pd.DataFrame(data=np.zeros((df.shape[0],31)),columns=['10_winsOpponent','10_winsChallanger','25_winsOpponent','25_winsChallanger','50_winsOpponent','50_winsChallanger','10_winRateOpponent','10_winRateChallanger','25_winRateOpponent','25_winRateChallanger','50_winRateOpponent','50_winRateChallanger','10_winLossRateOpponent','10_winLossRateChallanger','25_winLossRateOpponent','25_winLossRateChallanger','50_winLossRateOpponent','50_winLossRateChallanger','10_trackCountOpponent','10_trackCountChallanger','25_trackCountOpponent','25_trackCountChallanger','50_trackCountOpponent','50_trackCountChallanger','10_avgFuelConsumptionOnTrackOpponent','10_avgFuelConsumptionOnTrackChallenger','25_avgFuelConsumptionOnTrackOpponent','25_avgFuelConsumptionOnTrackChallenger','50_avgFuelConsumptionOnTrackOpponent','50_avgFuelConsumptionOnTrackChallenger','label'])


	for counter,row in df.iterrows():


		n = 10
		opponent = str(int(row['opponent']))
		challenger =  str(int(row['challenger']))
		date = row['race_driven']

		index = 0
		for index,_ in playerId_wins.iterrows():
			if(dates[index] == date):
				
				break

		if(n>index):
			if(index>0):
				n=index
			else:
				n=1

		


		if index-1-n<0:
			lower = 0
		else:
			lower = index-1-n

		

		winsOpponent = playerId_wins[opponent][lower:index].sum()
		winsOpponent = winsOpponent/n
		winsChallenger = playerId_wins[challenger][lower:index].sum()
		winsChallenger = winsChallenger/n


		featureDataFrame.at[counter,'10_winsOpponent'] = winsOpponent
		featureDataFrame.at[counter,'10_winsChallanger'] = winsChallenger

		
		winRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winRateOpponent = winRateOpponent/n
		winRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winRateChallenger = winRateChallenger/n
		featureDataFrame.at[counter,'10_winRateOpponent'] = winRateOpponent
		featureDataFrame.at[counter,'10_winRateChallanger'] = winRateChallenger

		winLossRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winLossRateOpponent = winLossRateOpponent/n
		winLossRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winLossRateChallenger = winLossRateChallenger/n	
		featureDataFrame.at[counter,'10_winLossRateOpponent'] = winLossRateOpponent
		featureDataFrame.at[counter,'10_winLossRateChallanger'] = winLossRateChallenger


		trackCountOpponent=0
		for t in range(lower,index):
			trackCountOpponent += playerId_trackCounts[opponent][t][int(row['track_id'])]
		trackCountOpponent = trackCountOpponent/n
		trackCountChallenger=0
		for t in range(lower,index):
			trackCountChallenger+=playerId_trackCounts[challenger][t][int(row['track_id'])]
		trackCountChallenger = trackCountChallenger/n
		featureDataFrame.at[counter,'10_trackCountOpponent'] = trackCountOpponent
		featureDataFrame.at[counter,'10_trackCountChallanger'] = trackCountChallenger
		avgFuelConsumptionOnTrackOpponent=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackOpponent += playerId_avgFuelCosumption[opponent][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackOpponent = avgFuelConsumptionOnTrackOpponent/n
		avgFuelConsumptionOnTrackChallenger=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackChallenger += playerId_avgFuelCosumption[challenger][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackChallenger = avgFuelConsumptionOnTrackChallenger/n
		featureDataFrame.at[counter,'10_avgFuelConsumptionOnTrackOpponent'] = avgFuelConsumptionOnTrackOpponent
		featureDataFrame.at[counter,'10_avgFuelConsumptionOnTrackChallenger'] =  avgFuelConsumptionOnTrackChallenger
		




		n = 25
		if(n>index):
			if(index>0):
				n=index
			else:
				n=1


		if index-1-n<0:
			lower = 0
		else:
			lower = index-1-n

		winsOpponent = playerId_wins[opponent][lower:index].sum()
		winsOpponent = winsOpponent/n
		winsChallenger = playerId_wins[challenger][lower:index].sum()
		winsChallenger = winsChallenger/n
		featureDataFrame.at[counter,'25_winsOpponent'] = winsOpponent
		featureDataFrame.at[counter,'25_winsChallanger'] = winsChallenger


		winRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winRateOpponent = winRateOpponent/n
		winRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winRateChallenger = winRateChallenger/n
		featureDataFrame.at[counter,'25_winRateOpponent'] = winRateOpponent
		featureDataFrame.at[counter,'25_winRateChallanger'] = winRateChallenger

		winLossRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winLossRateOpponent = winLossRateOpponent/n
		winLossRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winLossRateChallenger = winLossRateChallenger/n	
		featureDataFrame.at[counter,'25_winLossRateOpponent'] = winLossRateOpponent
		featureDataFrame.at[counter,'25_winLossRateChallanger'] = winLossRateChallenger

		
		trackCountOpponent=0
		for t in range(lower,index):
			trackCountOpponent += playerId_trackCounts[opponent][t][int(row['track_id'])]
		trackCountOpponent = trackCountOpponent/n
		trackCountChallenger=0
		for t in range(lower,index):
			trackCountChallenger+=playerId_trackCounts[challenger][t][int(row['track_id'])]
		trackCountChallenger = trackCountChallenger/n
		featureDataFrame.at[counter,'25_trackCountOpponent'] = trackCountOpponent
		featureDataFrame.at[counter,'25_trackCountChallanger'] = trackCountChallenger
		avgFuelConsumptionOnTrackOpponent=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackOpponent += playerId_avgFuelCosumption[opponent][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackOpponent = avgFuelConsumptionOnTrackOpponent/n
		avgFuelConsumptionOnTrackChallenger=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackChallenger += playerId_avgFuelCosumption[challenger][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackChallenger = avgFuelConsumptionOnTrackChallenger/n
		featureDataFrame.at[counter,'25_avgFuelConsumptionOnTrackOpponent'] = avgFuelConsumptionOnTrackOpponent
		featureDataFrame.at[counter,'25_avgFuelConsumptionOnTrackChallenger'] =  avgFuelConsumptionOnTrackChallenger
		

		n = 50
		if(n>index):
			if(index>0):
				n=index
			else:
				n=1


		if index-1-n<0:
			lower = 0
		else:
			lower = index-1-n

		winsOpponent = playerId_wins[opponent][lower:index].sum()
		winsOpponent = winsOpponent/n
		winsChallenger = playerId_wins[challenger][lower:index].sum()
		winsChallenger = winsChallenger/n
		featureDataFrame.at[counter,'50_winsOpponent'] = winsOpponent
		featureDataFrame.at[counter,'50_winsChallanger'] = winsChallenger


		winRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winRateOpponent = winRateOpponent/n
		winRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winRateChallenger = winRateChallenger/n
		featureDataFrame.at[counter,'50_winRateOpponent'] = winRateOpponent
		featureDataFrame.at[counter,'50_winRateChallanger'] = winRateChallenger

		winLossRateOpponent = playerId_winrate[opponent][lower:index].sum()
		winLossRateOpponent = winLossRateOpponent/n
		winLossRateChallenger = playerId_winrate[challenger][lower:index].sum()
		winLossRateChallenger = winLossRateChallenger/n	
		featureDataFrame.at[counter,'50_winLossRateOpponent'] = winLossRateOpponent
		featureDataFrame.at[counter,'50_winLossRateChallanger'] = winLossRateChallenger

		
		trackCountOpponent=0
		for t in range(lower,index):
			trackCountOpponent += playerId_trackCounts[opponent][t][int(row['track_id'])]
		trackCountOpponent = trackCountOpponent/n
		trackCountChallenger=0
		for t in range(lower,index):
			trackCountChallenger+=playerId_trackCounts[challenger][t][int(row['track_id'])]
		trackCountChallenger = trackCountChallenger/n
		featureDataFrame.at[counter,'50_trackCountOpponent'] = trackCountOpponent
		featureDataFrame.at[counter,'50_trackCountChallanger'] = trackCountChallenger
		avgFuelConsumptionOnTrackOpponent=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackOpponent += playerId_avgFuelCosumption[opponent][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackOpponent = avgFuelConsumptionOnTrackOpponent/n
		avgFuelConsumptionOnTrackChallenger=0
		for t in range(lower,index):
			avgFuelConsumptionOnTrackChallenger += playerId_avgFuelCosumption[challenger][t][int(row['track_id'])]
		avgFuelConsumptionOnTrackChallenger = avgFuelConsumptionOnTrackChallenger/n
		featureDataFrame.at[counter,'50_avgFuelConsumptionOnTrackOpponent'] = avgFuelConsumptionOnTrackOpponent
		featureDataFrame.at[counter,'50_avgFuelConsumptionOnTrackChallenger'] =  avgFuelConsumptionOnTrackChallenger
		


		if row['winner'] == row['opponent']:
			featureDataFrame.at[counter,'label'] = 1
		else:
			featureDataFrame.at[counter,'label'] = 0

		
		
		
	
	featureDataFrame.to_csv("features.csv",sep=";")



def plotCountTracks(df):
	# Plots race counts


	tracks = df['track_id'].to_numpy()
	uniques, counts	= np.unique(tracks,return_counts=True)

	plt.bar(uniques,counts)
	plt.xticks(uniques)
	plt.yscale('log')
	plt.xlabel('Track Id')
	plt.ylabel('Number of races (log scale)')
	plt.show()





def plotFuelConsumptionAgainstTrackID(df):
	#plots fuel_consumtion on every track

	df = df[df['status']=='finished']
	df = df[df['fuel_consumption']!='None']

	df['track_id'] = df['track_id'].astype('int32')
	df['fuel_consumption'] = df['fuel_consumption'].astype('float32')


	X1=[]
	Y1=[]
	
	for index, row in df.iterrows():
		if(row['status']=='finished'):
			if(str(row['track_id'])!=str(np.nan)):
				#print(str(row['fuel_consumption']))
				X1.append(int(row['track_id']))
				Y1.append(float(row['fuel_consumption']))
		

	plt.scatter(np.array(X1),np.array(Y1),color='blue',s=20)


	X3=[]
	Y3=[]
	x= np.sort(np.delete(df['track_id'].unique(),3))


	for a in x:
		Y3.append(df.loc[(df['track_id']==a)&(df['status']=='finished'),'fuel_consumption'].mean())
		X3.append(a)
		
	plt.xticks([3,4,5,6,7,8,9,10,11,12,13,14])


	plt.scatter(np.array(X3),np.array(Y3),color='blue',s=80,marker='X',edgecolors='black')
	plt.xlabel("Track Id")
	plt.ylabel("Fuel Consumption")
	plt.show()




def plotMoney(df):
	#plot histogram und history of the column money

	df['race_driven']=pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']=df['race_driven'].dt.date

	df = df[df['status']=='finished']
	

	dates = df['race_driven'].unique()
	dates = np.sort(dates)


	print("min Einsatz: "+str(df['money'].min()))
	print("max Einsatz: "+str(df['money'].max()))
	print("mean Einsatz: "+str(df['money'].mean()))
	print("median Einsatz: "+str(df['money'].median()))

	print("Races with Money greater 1000000 ")
	print(df[df['money']>1000000])

	avgMoney = []

	for d in dates:
		ds_date = df[df['race_driven']==d]
		avgMoney.append(ds_date['money'].sum()/ds_date.shape[0])
	plt.plot(dates,avgMoney)
	plt.yscale('log')
	plt.xlabel('Time')
	plt.ylabel('Average Money per Day')
	plt.show()


	plt.hist(df['money'],bins=30)
	plt.xlabel('Money')
	plt.ylabel('Number of races (log scale)')
	plt.yscale('log')
	plt.show()


def plotWeather(df):
	#plots weather

	uniques,counts = np.unique(df['weather'],return_counts=True)
	print(uniques)
	print(counts)
	plt.bar(uniques,counts)
	plt.show()


def plotNewPlayerRate(df,useLogScale=True):
	#plots new players per day

	df['race_created']=pd.to_datetime(dataFrame['race_created'],format='%d.%m.%Y')
	df['race_created']=df['race_created'].dt.date

	df = df[df['status']=='finished']
	

	dates = df['race_created'].unique()
	dates = np.sort(dates)

	players = []
	newPlayers = pd.DataFrame(np.zeros((dates.shape[0],2)),columns=['date','newPlayers'])

	newPlayers['date'] = dates

	for _,row in df.iterrows():
		#print(players)
		if(not row['challenger'] in players):
			newPlayers.loc[newPlayers['date']==row['race_created'],'newPlayers'] += 1
			players.append(row['challenger'])
		if(not row['opponent'] in players):
			newPlayers.loc[newPlayers['date']==row['race_created'],'newPlayers'] += 1
			players.append(row['opponent'])

	newPlayers['newPlayers'] = uniform_filter1d(newPlayers['newPlayers'],size=20)

	plt.plot(newPlayers['date'],newPlayers['newPlayers'])
	plt.xlabel('Time')

	if useLogScale:
		plt.ylabel('Number of new players per day (log)')
		plt.yscale('log')
	else:
		plt.ylabel('Number of new players per day (linear)')
		plt.yscale('linear')
	plt.show()






def plotPlayerNumRaces(df):
	#plots histogram of races driven by every player

	players = []
	for _,row in df.iterrows():
		players.append(row['opponent'])
		players.append(row['challenger'])


	uniques,counts = np.unique(np.array(players),return_counts=True)



	plt.hist(counts,bins=20)
	plt.yscale('log')
	plt.ylabel("Number of Players")
	plt.xlabel("Number of Races")
	plt.show()
	
def plotPlayDuration(df):
	#plots histogram of time between fist and last race

	players = []
	for _,row in df.iterrows():
		players.append(row['opponent'])
		players.append(row['challenger'])



	uniques = np.unique(np.array(players))

	df['race_driven']=pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']=df['race_driven'].dt.date


	durations = []



	maxD = 0
	maxDPlayer = 0
	maxFist = 0 
	maxLast = 0

	for p in uniques:
		tempDf = df.loc[(df['challenger']==p) | (df['opponent']==p)]

		fist = tempDf['race_driven'].min()
		last = tempDf['race_driven'].max()
		d = abs((last-fist).days)
		if d>maxD:
			maxD = d
			maxDPlayer = p
			maxFist = fist
			maxLast = last
		durations.append(abs((last-fist).days))

	
	print("Playing the longest is: " + str(maxDPlayer)+ "("+str(maxD)+" days)"+ " from "+str(maxFist) +" till "+str(maxLast))
	print("Mean Duration:" +str(np.mean(durations)))
	print("Median Duration:"+ str(np.median(durations)))



	plt.hist(durations,bins=20)
	plt.xlabel("Number of Days between fist and last race")
	plt.ylabel("Number of Players")
	plt.yscale('log')
	plt.show()



def plotDurationRaceNum(df):

	players = []
	for _,row in df.iterrows():
		players.append(row['opponent'])
		players.append(row['challenger'])


	uniques,counts = np.unique(np.array(players),return_counts=True)

	df['race_driven']=pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
	df['race_driven']=df['race_driven'].dt.date

	durations = []

	for p in uniques:
		tempDf = df.loc[(df['challenger']==p) | (df['opponent']==p)]

		fist = tempDf['race_driven'].min()
		last = tempDf['race_driven'].max()
		d = abs((last-fist).days)
		
		durations.append(abs((last-fist).days))


	plt.scatter(durations,counts)
	plt.xlabel("Days played")
	plt.ylabel("Races driven")


	plt.show()







if __name__ == "__main__":


	DATA_DIR = "./races.csv"
	pd.set_option('display.max_columns',None)
	dataFrame = pd.read_csv(DATA_DIR,sep=";",header=0)
	dataFrame['id'].astype('int32')
	dataFrame = dataFrame.set_index('id')
	dataFrame =dataFrame[0:100000]

	


	args = sys.argv
	if len(args)>1:
		if args[1] == 'testMARvsMCAR':
			testMARvsMCAR(dataFrame)
		elif args[1] == 'testFuelConsumptionCorrelation':
			testFuelConsumptionCorrelation(dataFrame)
		
		elif args[1] == 'imputeData':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			makeBrokenToNone(dataFrame)
			correctDate(dataFrame)
			imputeKNN(dataFrame)
		elif args[1] == 'createFeatures':
			getStats()
			createFeatures()
		elif args[1] == 'plotRaces':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			dataFrame['race_driven'] = pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')

			if len(args)>2:
				N=int(args[2])
			else:
				N=20
			plotRaces(dataFrame,N)
		elif args[1] == 'plotTopTrack':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			dataFrame['race_driven'] = pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')

			if len(args)>2:
				N=int(args[2])
			else:
				N=20
			if len(args)>3:

				b=bool(int(args[3]))
			else:
				b = False	

			plotTopTrack(dataFrame, N, b)
		elif args[1] == 'getAverageFuelConsumption':
			getAverageFuelConsumption(dataFrame)
		elif args[1] == 'plotPlayDuration':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			plotPlayDuration(dataFrame)
		elif args[1] == 'plotPlayerNumRaces':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			plotPlayerNumRaces(dataFrame)
		elif args[1] == 'plotNewPlayerRate':
			if len(args)>2:

				b=bool(int(args[2]))
			else:
				b = False	
			plotNewPlayerRate(dataFrame,b)
		elif args[1] == 'plotWeather':
			dataFrame = dataFrame[dataFrame['status']=='finished']
			plotWeather(dataFrame)
		elif args[1] == 'plotMoney':
			dataFrame = dataFrame[dataFrame['status']=='finished']
			dataFrame['race_driven'] = pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
			plotMoney(dataFrame)
		elif args[1] == 'plotFuelConsumptionAgainstTrackID':
			makeBrokenToNan(dataFrame)
			dataFrame.dropna()
			dataFrame = dataFrame[dataFrame['status']=='finished']
			dataFrame['race_driven'] = pd.to_datetime(dataFrame['race_driven'],format='%d.%m.%Y %H:%M')
			plotFuelConsumptionAgainstTrackID(dataFrame)

		elif args[1] == 'plotDurationRaceNum':
			dataFrame = dataFrame[dataFrame['opponent']!=0]
			dataFrame = dataFrame[dataFrame['status']=='finished']
			correctDate(dataFrame)
			plotDurationRaceNum(dataFrame)