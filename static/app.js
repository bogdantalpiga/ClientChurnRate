var myApp = angular.module('myApp', ['ui.router','homeController','dashboardController','chart.js']);

myApp.config(function ($stateProvider, $urlRouterProvider) {
  $urlRouterProvider.otherwise('/dashboard')
    // $stateProvider
    //     .state('home', {
    //         url: '/',
    //         templateUrl: 'static/partials/home.html',
    //         controller : 'homeController'
    //     })
    $stateProvider
        .state('dashboard',{
            url:'/dashboard',
            templateUrl:'static/partials/dashboard.html',
            controller: 'dashboardController'
        })

});