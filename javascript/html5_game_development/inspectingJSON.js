
// This is an example of how the JSON would be structured.
// Note that chaingun_impact.png is not here.
//
// Note that this is an actual Javascript object, whereas
// JSON is a string that represents that object.

JSONExample = {
    "frames": {
        "chaingun.png": {
            "frame": {
                "x": 1766,
                "y": 202,
                "w": 42,
                "h": 34
            },
            "rotated": false,
            "trimmed": true,
            "spriteSourceSize": {
                "x": 38,
                "y": 32,
                "w": 42,
                "h": 34
            },
            "sourceSize": {
                "w": 128,
                "h": 128
            }
        },
        "chaingun_impact.png": {
            "frame": {
                "x":1162,
                "y":322,
                "w":38,
                "h":34},
            "rotated": false,
            "trimmed": true,
            "spriteSourceSize": {
                "x":110,
                "y":111,
                "w":38,
                "h":34},
            "sourceSize": {
                "w":256,
                "h":256}
        },
        "chaingun_impact_0000.png": {
            "frame": {
                "x": 494,
                "y": 260,
                "w": 22,
                "h": 22
            },
            "rotated": false,
            "trimmed": true,
            "spriteSourceSize": {
                "x": 113,
                "y": 108,
                "w": 22,
                "h": 22
            },
            "sourceSize": {
                "w": 256,
                "h": 256
            }
        }
    }
};

// The above is an example of how the JSON would be structured.
// Note that chaingun_impact.png is not here, we'll call your
// parseJSON function with the full JSON input.
//
// Note also that the above is an actual Javascript object, whereas
// JSON is a string that represents that object.
parseJSON = function (weaponJSON) {
    // First, use the JSON.parse function to
    // parse the passed in weaponJSON.
    //
    // Next, grab the 'x' data field within
    // 'spriteSourceSize' of 'chaingun_impact.png'
    //
    // After that, print this value to the console
    // and also return it.
    // YOUR CODE HERE
    var parsedJSON = JSON.parse(weaponJSON);
    console.log(parsedJSON['frames']['chaingun_impact.png']['spriteSourceSize']['x']);
    
};

