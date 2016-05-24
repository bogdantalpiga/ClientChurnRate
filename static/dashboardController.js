var app=angular.module('dashboardController',['ui.router','chart.js'])

var SERVER='http://localhost:5000'
app.controller('dashboardController',function($scope,$http,$state,$rootScope){
	$scope.variabiles={};
	$scope.panel=1;
	$scope.matrix={};

	$scope.demoSet=[];
	
  $scope.labels = ['2006', '2007', '2008', '2009', '2010', '2011', '2012'];
  $scope.series = ['Series A', 'Series B'];

  $scope.data = [
    [65, 59, 80, 81, 56, 55, 40],
    [28, 48, 40, 19, 86, 27, 90]
  ];

	$scope.predit=function(){
		//"tenure":33,"MonthlyCharges":88.9499969482,"TotalCharges":3027.6499023438,"gender_converted":1,"SeniorCitizen_converted":0,
		//"Partner_converted":1,"Dependents_converted":0,"PhoneService_converted":1,"PaperlessBilling_converted":0,
		//"MultipleLines_No":1.0,"MultipleLines_No phone service":0.0,"MultipleLines_Yes":0.0,"InternetService_DSL":0.0,
		//"InternetService_Fiber optic":1.0,"InternetService_No":0.0,"OnlineSecurity_No":0.0,"OnlineSecurity_No internet service":0.0,
		//"OnlineSecurity_Yes":1.0,"OnlineBackup_No":1.0,"OnlineBackup_No internet service":0.0,"OnlineBackup_Yes":0.0,
		//"DeviceProtection_No":1.0,"DeviceProtection_No internet service":0.0,"DeviceProtection_Yes":0.0,"TechSupport_No":0.0,
		//"TechSupport_No internet service":0.0,"TechSupport_Yes":1.0,"StreamingTV_No":0.0,"StreamingTV_No internet service":0.0,
		//"StreamingTV_Yes":1.0,"StreamingMovies_No":1.0,"StreamingMovies_No internet service":0.0,"StreamingMovies_Yes":0.0,
		//"Contract_Month-to-month":0.0,"Contract_One year":0.0,"Contract_Two year":1.0,"PaymentMethod_Bank transfer (automatic)":0.0,
		//"PaymentMethod_Credit card (automatic)":0.0,"PaymentMethod_Electronic check":1.0,"PaymentMethod_Mailed check":0.0
	}
	$scope.getVariabiles=function(){
		console.log('get variables')
		$http.get(SERVER+"/variabiles")
		.then(function(rsp){
			console.log('got variables')
			$scope.variabiles=rsp.data;
			var gbm=$scope.variabiles.gbm;
			var forest=$scope.variabiles.randomForestTree;
			$scope.labels=[];
			$scope.data=[];
			gbmData=[];
			forestData=[];
			for(var key in gbm){
				$scope.labels.push(key);
				gbmData.push(gbm[key])
				forestData.push(forest[key])
			}
			$scope.data.push(gbmData);
			$scope.data.push(forestData);
			$scope.series=['Gradient boosting','Random Forest']
			$scope.panel=2
		})
		.catch(function(exp){
			console.warn(exp);
		})
	}
	$scope.analyse=function(){
		console.log('get matrix')
		$http.get(SERVER+'/getMAtrix')
		.then(function(rsp){
			$scope.matrix=rsp.data;
			console.log('got matrix');
			var gbm=$scope.matrix.Gradient;
			var forest=$scope.matrix.RandomTree;
			createPerformanceChart(gbm,forest);
			

			$scope.panel=3;
		})
		.catch(function(exp){
			console.warn(exp);
		})
	}

	function createPerformanceChart(gbm,forest){
		$scope.labels=['Accuracy','Specificity','Sensitivity','Precision']
		$scope.series=['Gradient boosting','Random Forest']
		$scope.data=[];
		perfGbm=[];
		perfForest=[];
		perfGbm.push(accuracy(gbm));perfGbm.push(specificity(gbm));perfGbm.push((sensitivity(gbm)));perfGbm.push((precision(gbm)));
		perfForest.push(accuracy(forest));perfForest.push(specificity(forest));perfForest.push((sensitivity(forest)));
		perfForest.push((precision(forest)));
		$scope.data.push(perfGbm);
		$scope.data.push(perfForest)

	}
	function accuracy(obj){
		return (obj.true_positive+obj.true_negative)/(obj.true_negative+obj.true_positive+obj.false_positive+obj.false_negative)
	}
	function specificity(obj){
		return obj.true_negative/(obj.true_negative+obj.false_positive)
	}
	function sensitivity(obj){
		return obj.true_positive/(obj.true_positive+obj.false_negative)
	}
	function precision(obj){
		return obj.true_positive/(obj.true_positive+obj.false_positive)
	}

	$scope.getDemoSet=function(){
		console.log('getDemoSet')
		$http.get(SERVER+'/demoSet')
		.then(function(rsp){
			console.log('Got demo set')
			$scope.demoSet=rsp.data;
			for(var i=0;i<$scope.demoSet.length;i++){
				$scope.demoSet[i].checked=false;
				$scope.demoSet[i].predictedChurn=-1;
			}
			$scope.panel=4;
		})
		.catch(function(exp){
			console.warn(exp);
		})
	}

	$scope.predict=function(){
		var payload=[];
		console.log("Predicting")
		var clicked=[]
		for(var i=0;i<$scope.demoSet.length;i++){
			if($scope.demoSet[i].checked===true){
				console.log(i);
				clicked.push(i);
				//clonez obiectul din model
				var newObject = JSON.parse(JSON.stringify($scope.demoSet[i]));
				delete newObject['checked']
				payload.push(newObject)
				
			}
		}
		$http.post(SERVER+"/predict",payload)
				.then(function(rsp){
					console.log(rsp.data)
					var contor=0;
					for(var i=0;i<clicked.length;i++){
						$scope.demoSet[clicked[i]].predictedChurn=rsp.data[contor];
						contor++; 
					}
				})
				.catch(function (exp){
					console.warn(exp);
				})
	}

	$scope.recalc=function(){
		$http.get(SERVER+'/newModel')
		.then(function(rsp){
			alert(rsp.data.Success)
		})
		.catch(function (exp){
			console.warn(exp);
		})
	}
window.sc=$scope;
})
