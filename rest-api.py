#!flask/bin/python
import churnPredict as cp
from flask import Flask,jsonify,request, Response
import pandas as pd
import json
from collections import OrderedDict
# from flask.ext.login import LoginManager, UserMixin, login_required
# from flask.ext.bcrypt import Bcrypt
from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp
from flask.ext.mysql import MySQL
from shutil import copyfile
import datetime

mysql = MySQL()


class User(object):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __str__(self):
        return "User(id='%s')" % self.id

users = [
    User(1, 'user1', 'abcxyz'),
    User(2, 'user2', 'abcxyz'),
]

username_table = {u.username: u for u in users}
userid_table = {u.id: u for u in users}

def authenticate(username, password):
    user = username_table.get(username, None)
    if user and safe_str_cmp(user.password.encode('utf-8'), password.encode('utf-8')):
        return user

def identity(payload):
    user_id = payload['identity']
    return userid_table.get(user_id, None)

app = Flask(__name__)
jwt = JWT(app, authenticate, identity)
app.config["SECRET_KEY"] = "MYSECRET"
# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '1234567_8'
app.config['MYSQL_DATABASE_DB'] = 'digital_services'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
#######################
rez=[]

varList=['customerID','Churn_converted','tenure', 
	'MonthlyCharges', 'TotalCharges', 'gender_converted', 
	'SeniorCitizen_converted', 'Partner_converted', 
	'Dependents_converted', 'PhoneService_converted', 
	'PaperlessBilling_converted', 'MultipleLines_No', 
	'MultipleLines_No phone service', 'MultipleLines_Yes', 
	'InternetService_DSL', 'InternetService_Fiber optic', 
	'InternetService_No', 'OnlineSecurity_No', 
	'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
	'OnlineBackup_No', 'OnlineBackup_No internet service', 
	'OnlineBackup_Yes', 'DeviceProtection_No', 
	'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
	'TechSupport_No', 'TechSupport_No internet service', 
	'TechSupport_Yes', 'StreamingTV_No', 
	'StreamingTV_No internet service', 'StreamingTV_Yes', 
	'StreamingMovies_No', 'StreamingMovies_No internet service', 
	'StreamingMovies_Yes', 'Contract_Month-to-month', 
	'Contract_One year', 'Contract_Two year', 
	'PaymentMethod_Bank transfer (automatic)', 
	'PaymentMethod_Credit card (automatic)', 
	'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

header="customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn"



#########################################################   

@app.route('/')
def index():
    return app.send_static_file('index.html') 

# @app.route('/login')
# @jwt_required()
# def checkUser():
# 	return json.dumps({"success":"true"})

@app.route('/newModel',methods=['GET'])
def createNewModel():
	exportDatas()
	global rez
	rez=cp.centralFlow()
	response={"Success":"Both models were recalculated"}
	return json.dumps(response)

def exportDatas():
	conn=mysql.connect()
	cursor=conn.cursor()
	
	now = datetime.datetime.now()
	filename='exportedDB_'+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+".csv"
	print("File name ="+str(filename))
	cursor.callproc('sp_exportDB',(filename,))
	copyfile('/tmp/'+filename,'/home/bogdan/Desktop/licenta/datas.csv')
	with open('/home/bogdan/Desktop/licenta/datas.csv', 'r+') as f:
		content = f.read()
		f.seek(0, 0)
		f.write(header + '\n' + content)


@app.route('/getMAtrix',methods=['GET'])
# @jwt_required()
def getMatrix():
	gbmM=rez[5]
	randomM=rez[4]
	obj={}
	obj["RandomTree"]=randomM
	obj["Gradient"]=gbmM
	return json.dumps(obj)

@app.route('/scoring', methods=['GET'])
def getScores():
    return rez[3]

@app.route('/predict', methods=['POST'])
def predictForest():
	# js=[{"customerID":"4622-YNKIJ","Churn_converted":0,"tenure":33,"MonthlyCharges":88.9499969482,"TotalCharges":3027.6499023438,"gender_converted":1,"SeniorCitizen_converted":0,"Partner_converted":1,"Dependents_converted":0,"PhoneService_converted":1,"PaperlessBilling_converted":0,"MultipleLines_No":1.0,"MultipleLines_No phone service":0.0,"MultipleLines_Yes":0.0,"InternetService_DSL":0.0,"InternetService_Fiber optic":1.0,"InternetService_No":0.0,"OnlineSecurity_No":0.0,"OnlineSecurity_No internet service":0.0,"OnlineSecurity_Yes":1.0,"OnlineBackup_No":1.0,"OnlineBackup_No internet service":0.0,"OnlineBackup_Yes":0.0,"DeviceProtection_No":1.0,"DeviceProtection_No internet service":0.0,"DeviceProtection_Yes":0.0,"TechSupport_No":0.0,"TechSupport_No internet service":0.0,"TechSupport_Yes":1.0,"StreamingTV_No":0.0,"StreamingTV_No internet service":0.0,"StreamingTV_Yes":1.0,"StreamingMovies_No":1.0,"StreamingMovies_No internet service":0.0,"StreamingMovies_Yes":0.0,"Contract_Month-to-month":0.0,"Contract_One year":0.0,"Contract_Two year":1.0,"PaymentMethod_Bank transfer (automatic)":0.0,"PaymentMethod_Credit card (automatic)":0.0,"PaymentMethod_Electronic check":1.0,"PaymentMethod_Mailed check":0.0},
	# {"customerID":"4622-YNKIJ","Churn_converted":0,"tenure":33,"MonthlyCharges":88.9499969482,"TotalCharges":3027.6499023438,"gender_converted":1,"SeniorCitizen_converted":0,"Partner_converted":1,"Dependents_converted":0,"PhoneService_converted":1,"PaperlessBilling_converted":0,"MultipleLines_No":1.0,"MultipleLines_No phone service":0.0,"MultipleLines_Yes":0.0,"InternetService_DSL":0.0,"InternetService_Fiber optic":1.0,"InternetService_No":0.0,"OnlineSecurity_No":0.0,"OnlineSecurity_No internet service":0.0,"OnlineSecurity_Yes":1.0,"OnlineBackup_No":1.0,"OnlineBackup_No internet service":0.0,"OnlineBackup_Yes":0.0,"DeviceProtection_No":1.0,"DeviceProtection_No internet service":0.0,"DeviceProtection_Yes":0.0,"TechSupport_No":0.0,"TechSupport_No internet service":0.0,"TechSupport_Yes":1.0,"StreamingTV_No":0.0,"StreamingTV_No internet service":0.0,"StreamingTV_Yes":1.0,"StreamingMovies_No":1.0,"StreamingMovies_No internet service":0.0,"StreamingMovies_Yes":0.0,"Contract_Month-to-month":0.0,"Contract_One year":0.0,"Contract_Two year":1.0,"PaymentMethod_Bank transfer (automatic)":0.0,"PaymentMethod_Credit card (automatic)":0.0,"PaymentMethod_Electronic check":1.0,"PaymentMethod_Mailed check":0.0}]
	varListRaw=["customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines",
	"InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
	"StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"]

	js = request.json
	print(js)
	
	df=pd.read_json(json.dumps(js))#,typ='series') cand e doar unul, si frame cand sunt mai multe
	df=df[varListRaw] #TODO
	print("############")
	print(df)
	df=cp.cleanSingularData(df)
	print(df)
	fr_pred=cp.predictRandomTree(df,rez[0])
	print('PAS 6')
	count=0
	fr_json=[]
	for pred in fr_pred:
		fr_json.append(pred)#["Predicted "+str(count)]=pred
		count=count+1
	print(fr_json)
	return Response(json.dumps(fr_json),  mimetype='application/json')

@app.route('/demoSet',methods=['GET'])
def getDemoSet():
	df=cp.getDemoSet()
	return df.to_json(orient='records')

@app.route('/variabiles',methods=['GET'])
def getVariables():
	gbm_feature_importance=rez[1].feature_importances_
	poz=2

	random_feature_importance=rez[0].feature_importances_
	gbmImportance={}
	randomImportance={}
	importanta={}
	#ne trebuie si o lista cu toate numele variabilelor
	for valoare in gbm_feature_importance:
		gbmImportance[varList[poz]]=valoare
		poz=poz+1
	poz=2
	for valoare in random_feature_importance:
		randomImportance[varList[poz]]=valoare
		poz=poz+1
	#print(gbm_feature_importance)
	importanta['randomForestTree']=randomImportance
	importanta['gbm']=gbmImportance
	return jsonify(importanta)

if __name__ == '__main__':
	rez2=cp.centralFlow()
	#print(rez2[3])
	rez=rez2
	app.run(debug=True)
	