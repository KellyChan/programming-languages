
parseJSON = function (weaponJSON) {
    parsedJSON = JSON.parse(weaponJSON);

    return parsedJSON['frames']['chaingun_impact.png']['spriteSourceSize']['x'];
};

var setup = function() {
// Create a new XMLHttpRequest object
//
// YOUR CODE HERE


// then use its open method to to define the request that
// will be sent. The parameters to 'open' are:
//
// 1) The HTTP method to use, in our case we want
//    "GET".
// 2) The resource to request, in this case we're
//    interested in "/media/resources/gamedev/weapon.json".
// 3) A boolean indicating whether or not we want
//    the request to be asynchronous or not. True
//    means we do want it to be asynchronous.
//
// YOUR CODE HERE

  
  
// After that, we want to define the onload method
// of the request to be our parsing function from
// before. We've included that code above for
// reference. A few things to keep in mind here:
//
// 1) This function can't take any parameters.
// 2) Instead of parsing 'weaponJSON', we'll need
//    to parse the 'responseText' member of the
//    request object.
// 3) You can access the request object inside
//    your 'onload' function by using the 'this'
//    keyword.

// YOUR CODE HERE



// Finally, we want to call the send method of the
// request object to actually send the request.
//
// YOUR CODE HERE
    var xhr = new XMLHttpRequest();
    
    xhr.open("GET", "/media/resources/gamedev/weapon.json", true);
    
    xhr.onload = function(){
        var parsedJSON = JSON.parse(this.responseText);
        return parsedJSON['frames']['chaingun_impact.png']['spriteSourceSize']['x'];
    };
    
    xhr.send();   
    
};
