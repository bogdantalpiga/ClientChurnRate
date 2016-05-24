var app=angular.module('homeController',['ui.router'])

var SERVER='http://localhost:5000'

app.controller("homeController",function($scope,$http,$state,$rootScope){
    $scope.username="";
    $scope.password="";
    $scope.token="";
    $scope.token;
    $scope.authenticate=function(){
        $http.post(SERVER+"/auth",{username:$scope.username,password:$scope.password})
        .then(function(resp){
           console.log(resp.data);
           var token=resp.data.token;
           $rootScope.token=token;//TODO aici o sa salvez token-ul
           $scope.token=token;
           $state.go("dashboard")
        })
        .catch(function(exp){
            console.log(exp);
        })
    }
    window.sc=$scope;
})