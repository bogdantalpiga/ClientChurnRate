import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import assert_all_finite
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

my_dict={}

def cleanData( adresa ):

	df = pd.read_csv(adresa)
	df.info()
	
	cols=df.columns.tolist()
	for column in cols:
		if df[column].unique().size == 2:
			df[column+'_converted']=df[column].map({df[column].unique()[0]:0,
			df[column].unique()[1]:1}).astype(int)
			print ('Convert '+column+' to 0='+str(df[column].unique()[0])+' and 1='+str(df[column].unique()[1]))
			obj={}
			obj['0']=df[column].unique()[0]
			obj['1']=df[column].unique()[1]
			my_dict[column]=obj
			df=df.drop(column,axis=1)

	dummyColumns=['MultipleLines','InternetService','OnlineSecurity',
				'OnlineBackup','DeviceProtection','TechSupport',
				'StreamingTV','StreamingMovies','Contract','PaymentMethod']
	for column in dummyColumns:
		
		df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
		print ('Convert '+column+' to dummy column')
		df=df.drop(column,axis=1)

	df['TotalCharges']=df['TotalCharges'].convert_objects(convert_numeric=True)
	columns=df.columns.tolist();

	for column in columns:
		if df[column].dtype==np.float64:
			df[column]=df[column].astype('float32')
		if df[column].dtype==np.int64:
			df[column]=df[column].astype('int32')

	columns=[columns[0]]+[columns[10]]+columns[1:10]+columns[11:]
	#print columns
	df=df[columns]
	df['TotalCharges'] = df[['TotalCharges', 'MonthlyCharges']].apply(lambda x:
						x['MonthlyCharges'] 
							if pd.isnull(x['TotalCharges'])
							else x['TotalCharges'], axis=1
						)
	print(df.columns.tolist())
	df.info()
	return df

def modelRandomTree(df_train):
	train_values=df_train.values
	model=RandomForestClassifier(n_estimators=15,max_features=0.1,max_depth=10,min_samples_leaf=15)
	model=model.fit(train_values[0:,2:],np.array(train_values[0:,1]).astype(int))
	return model

def predictRandomTree(df_test,model):
	test_values=df_test.values
	print('PAS 1')
	if test_values.ndim==2:
		print('PAS 2')
		rez=model.predict(test_values[:,2:])
		print('PAS 3')
	else:
		print('PAS 4')
		rez=model.predict(test_values[2:])
		print('PAS 5')
	return rez

def predictionRate(df_result,count):
	falseNegative=0
	falsePositive=0
	truePositive=0
	trueNegative=0
	values=df_result.values
	for arr in values:
		if arr[1]==0 and arr[2]==1:
			falsePositive=falsePositive+1
		elif arr[1]==1 and arr[2]==0:
			falseNegative=falseNegative+1
		elif arr[1]==0 and arr[2]==0:
			trueNegative=trueNegative+1
		else :
			truePositive=truePositive+1
	
	return [truePositive,trueNegative,falsePositive,falseNegative]

def modelGradient(df_train):
	train_values=df_train.values
	model=GradientBoostingClassifier(n_estimators=50,max_depth=5,min_samples_leaf=10,max_features=0.1)
	model.fit(train_values[0:,2:],np.array(train_values[0:,1]).astype(int))
	return model

def predictGradient(df_test,model):
	test_values=df_test.values
	rez=model.predict(test_values[:,2:])
	return rez

def tunningGradient(df_train):
	params={'n_estimators':[5,10,15,50],
			'learning_rate':[0.01,0.1,0.5,1],
			'max_depth':[5,10],
			'min_samples_leaf':[5,10,15,20],
			'max_features':[1.0,0.3,0.1]}
	estimator=GradientBoostingClassifier(n_estimators=50)
	train_values=df_train.values
	greed_search=GridSearchCV(estimator,params).fit(train_values[0:,2:],np.array(train_values[0:,1]).astype(int))
	return greed_search.best_params_
def tunningRandomTree(df_train):
	params={'n_estimators':[5,10,15,50],
			
			'max_depth':[5,10],
			'min_samples_leaf':[5,10,15,20],
			'max_features':[1.0,0.3,0.1]}
	estimator=RandomForestClassifier(n_estimators=50)
	train_values=df_train.values
	greed_search=GridSearchCV(estimator,params).fit(train_values[0:,2:],np.array(train_values[0:,1]).astype(int))
	return greed_search.best_params_

def blendModels(classifiers,X,X_test,y,crossVld,dataset_blend_test):
	if crossVld==False:
		dataset_blend_train = np.zeros((X.shape[0], len(classifiers)))
		#dataset_blend_test = np.zeros((X_test.shape[0], len(classifiers)))
		for j, clf in enumerate(classifiers):
			#print j, clf
			clf.fit(X, y)
			dataset_blend_train[:,j]=clf.predict_proba(X)[:,1]
			dataset_blend_test[:,j]=clf.predict_proba(X_test)[:,1]
		logisticModel=LogisticRegression()
		logisticModel.fit(dataset_blend_train,y)
		return logisticModel
	else: 
		dataset_blend_train = np.zeros((X.shape[0], len(classifiers)))
		skf = list(StratifiedKFold(y, 10))
		for j, clf in enumerate(classifiers):
			dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
			for i, (train, test) in enumerate(skf):
				#print "Fold", i
				X_train = X[train]
				y_train = y[train]
				X_t = X[test]
				y_t = y[test]
				clf.fit(X_train, y_train)
				y_submission = clf.predict_proba(X_t)[:,1]
				dataset_blend_train[test, j] = y_submission
				dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
				dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
		logisticModel=LogisticRegression()
		logisticModel.fit(dataset_blend_train,y)
		return logisticModel


def getConfusionMatrix(predicted_output,real_output):
	obj={"true_positive":0,
		 "false_positive":0,
		 "false_negative":0,
		 "true_negative":0}
	
	for k in range(len(predicted_output)):
		i=predicted_output[k]
		j=real_output[k]
		print(str(i)+" : "+str(j))
		if (i==j and j==0):
			obj['true_negative']=obj['true_negative']+1
		if(i==j and j==1):
			obj['true_positive']=obj['true_positive']+1
		if(i!=j and i==0):
			obj['false_negative']=obj['false_negative']+1
		if(i!=j and i==1):
			obj['false_positive']=obj['false_positive']+1

	print("ConfusionMatrix:")
	print(obj)
	return obj

def getDemoSet():
	df = pd.read_csv('datas.csv')
	return df[2063:2113]

def cleanSingularData(df):
	cols=df.columns.tolist()
	controllList=['Churn','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
	for column in controllList:
		# if df[column]==df_model_initial[column]:
		# 	df[column+'_converted']=df_model[column+'_converted'].astype(int)
		# elif df_model[column+'_converted']==0:
		# 	df[column+'_converted']=1
		# else:
		# 	df[column+'_converted']=0

		df[column+'_converted']=df[column].map({my_dict[column]['0']:0,
		my_dict[column]['1']:1}).astype(int)
		
		df=df.drop(column,axis=1)

	dummyColumns=['MultipleLines','InternetService','OnlineSecurity',
				'OnlineBackup','DeviceProtection','TechSupport',
				'StreamingTV','StreamingMovies','Contract','PaymentMethod']
	resultColumns=["MultipleLines_No","MultipleLines_No phone service","MultipleLines_Yes","InternetService_DSL",
"InternetService_Fiber optic","InternetService_No","OnlineSecurity_No","OnlineSecurity_No internet service",
"OnlineSecurity_Yes","OnlineBackup_No","OnlineBackup_No internet service","OnlineBackup_Yes",
"DeviceProtection_No","DeviceProtection_No internet service","DeviceProtection_Yes","TechSupport_No",
"TechSupport_No internet service","TechSupport_Yes","StreamingTV_No","StreamingTV_No internet service",
"StreamingTV_Yes","StreamingMovies_No","StreamingMovies_No internet service","StreamingMovies_Yes",
"Contract_Month-to-month","Contract_One year","Contract_Two year","PaymentMethod_Bank transfer (automatic)",
"PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check"]
	for resultColumn in resultColumns:
		df[resultColumn]=0

	for index,row in df.iterrows():
		for column in dummyColumns:
			print("index="+str(index)+" column=1 este "+column+"_"+row[column])
			df[column+"_"+row[column]][index]=1

	for column in dummyColumns:
		#df[dummyColumn].map({dummyColumn+" "+df[dummyColumn]:1})	
		#df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
		print ('Convert '+column+' to dummy column')
		df=df.drop(column,axis=1)

	df['TotalCharges']=df['TotalCharges'].convert_objects(convert_numeric=True)
	columns=df.columns.tolist();

	for column in columns:
		if df[column].dtype==np.float64:
			df[column]=df[column].astype('float32')
		if df[column].dtype==np.int64:
			df[column]=df[column].astype('int32')

	columns=[columns[0]]+[columns[10]]+columns[1:10]+columns[11:]
	#print columns
	df=df[columns]
	df['TotalCharges'] = df[['TotalCharges', 'MonthlyCharges']].apply(lambda x:
						x['MonthlyCharges'] 
							if pd.isnull(x['TotalCharges'])
							else x['TotalCharges'], axis=1
						)
	print(df.columns.tolist())
	df.info()
	return df


def centralFlow():
	df=cleanData('datas.csv')
	df_train=df[2113:] #swap
	df_test=df[:2063]
	
	# df_model=df.ix[0]
	# controllList=['Churn','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
	# for column in controllList:
	# 	#obj={"after":df[0][column]}
	# 	my_dict[column][df[column]]=df.ix[0][column]


	#df_demo=df[2063:2113]
	#print('>>>>>>> JSON DataFrame')
	#print(df_test.iloc[[125]].reset_index().to_json(orient='records'))
	modelR=modelRandomTree(df_train)
	#Forest_model=modelR
	output=predictRandomTree(df_test,modelR)
	confusionRandomTree=getConfusionMatrix(output.astype(int),df_test['Churn_converted'].astype(int))
	result = np.c_[df_test['customerID'], 
					df_test['Churn_converted'].astype(int),
					output.astype(int)]
	columns=['customer_id','churn','Forest_churn']
	count=len(df_test)
	index=range(1,count+1) #pentru header
	df_result=pd.DataFrame(result,index=index,columns=columns)
	df_result.to_csv('Forest_churn_result.csv')
	rating=predictionRate(df_result,count)
	rate=(float(rating[0]+rating[1])/float (count))*100####TODO de gasit o modalitate de rating mai buna
	print('Rata corectitudinii pentru ForestTree este de: '+str(rate)+'%')
	s='Rata corectitudinii pentru ForestTree este de: '+str(rate)+'%'

	modelG=modelGradient(df_train)
	#GBM_model=modelG
	output=predictGradient(df_test,modelG)
	confusionGBM=getConfusionMatrix(output.astype(int),df_test['Churn_converted'].astype(int))
	result = np.c_[df_test['customerID'], 
					df_test['Churn_converted'].astype(int),
					output.astype(int)]
	columns=['customer_id','churn','GBM_churn']
	count=len(df_test)
	index=range(1,count+1) #pentru header
	df_result=pd.DataFrame(result,index=index,columns=columns)
	df_result.to_csv('GBM_churn_result.csv')
	rating=predictionRate(df_result,count)
	rate=(float(rating[0]+rating[1])/float (count))*100####TODO de gasit o modalitate de rating mai buna
	print('Rata corectitudinii pentru GradientBoosting este de: '+str(rate)+'%')
	s=s+' \n'+'Rata corectitudinii pentru GradientBoosting este de: '+str(rate)+'%'

	X_test=df_test.values[0:,2:]
	classifiers=[modelR,modelG]
	dataset_blend_test = np.zeros((X_test.shape[0], len(classifiers)))
	blendModel=blendModels(classifiers,df_train.values[0:,2:],X_test,
		np.array(df_train.values[0:,1]).astype(int),False,dataset_blend_test)
	y_blend=blendModel.predict(dataset_blend_test)
	result = np.c_[df_test['customerID'], 
					df_test['Churn_converted'].astype(int),
					y_blend.astype(int)]
	df_result=pd.DataFrame(result,index=index,columns=columns)
	df_result.to_csv('Blended_churn_w_crossVld.csv')
	rating=predictionRate(df_result,count)
	rate=(float(rating[0]+rating[1])/float (count))*100
	print('Rata corectitudinii pentru Blending este de: '+str(rate)+'%')
	s=s+'\n'+'Rata corectitudinii pentru Blending este de: '+str(rate)+'%'

	blendModel=blendModels(classifiers,df_train.values[0:,2:],X_test,
		np.array(df_train.values[0:,1]).astype(int),True,dataset_blend_test)
	#logistic_model=blendModel
	y_blend=blendModel.predict(dataset_blend_test)
	result = np.c_[df_test['customerID'], 
					df_test['Churn_converted'].astype(int),
					y_blend.astype(int)]
	df_result=pd.DataFrame(result,index=index,columns=columns)
	df_result.to_csv('Blended_churn_crossVld.csv')
	rating=predictionRate(df_result,count)
	rate=(float(rating[0]+rating[1])/float (count))*100
	print('Rata corectitudinii pentru Blending cu cross validation este de: '+str(rate)+'%')
	s=s+'\n'+'Rata corectitudinii pentru Blending cu cross validation este de: '+str(rate)+'%'
	#train_values[0:,2:],np.array(train_values[0:,1]).astype(int)
	#print('------------ tunning RandTree')
	#print tunningRandomTree(df_train)
	rez=[modelR,modelG,blendModel,s,confusionRandomTree,confusionGBM]
	return rez






