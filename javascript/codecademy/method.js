var bob = new Object();
bob.age = 30;
// this time we have added a method, setAge
bob.setAge = function (newAge){
  bob.age = newAge;
};

bob.getYearOfBirth = function () {
  return 2013 - bob.age;
};
console.log(bob.getYearOfBirth());
