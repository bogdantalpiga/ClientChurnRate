var myApp = angular.module('myApp', ['ngRoute']);

myApp.config(function ($routeProvider) {
  $routeProvider
    .when('/', {
      templateUrl: 'partials/home.html',
      
    })
    // .when('/login', {
    //   templateUrl: 'partials/login.html',
    //   controller: 'loginController',
    //   access: {restricted: false}
    // })
    // .when('/logout', {
    //   controller: 'logoutController',
    //   access: {restricted: true}
    // })
    // .when('/register', {
    //   templateUrl: 'partials/register.html',
    //   controller: 'registerController',
    //   access: {restricted: false}
    // })
    .when('/one', {
      template: '<h1>This is page one!</h1>',
      
    })
    .when('/two', {
      template: '<h1>This is page two!</h1>',
      
    })
    .otherwise({
      redirectTo: '/'
    });
});