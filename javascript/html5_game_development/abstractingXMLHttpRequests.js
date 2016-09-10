

// Fill out the provided xhrGet function to abstract
// out the functionality of performing an XMLHttpRequest
// GET request.
//
// The provided parameters are the URI to make the request
// to, the callback to call at onload, and the responseType,
// if necessary. If we don't need a special responseType,
// assume that that parameter is null.
//
// Now, we're going to assume that the callback takes the
// request object as a parameter, instead of taking no
// parameters.
//
// We've provided you modified versions of the previous
// two callbacks below. At the bottom of the file, we call
// xhrGet with both callbacks to help you with testing your
// code.

function xhrGet(reqUri, callback, type) {
    // YOUR CODE HERE
        var caller = xhrGet.caller;
    
        var xhr = new XMLHttpRequest();
        xhr.open("GET", reqUri, true);

        if (type) {
            xhr.responseType = type;
        }

        xhr.onload = function () {
            if (callback) {
                try {
                    callback(xhr);
                } catch(e) {
                    throw 'xhrGet failed:\n' + reqUri + '\nException: ' + e + '\nresponseText: ' + xhr.responseText + '\ncaller: ' + caller;
                }
            }
        };
    
        xhr.send();
}

parseJSON = function (xhr) {
    var parsedJSON = JSON.parse(xhr.responseText);
    return parsedJSON['frames']['chaingun_impact.png']['spriteSourceSize']['x'];

};

playSound = function (xhr) {
    try {
        var context = new webkitAudioContext();

        var mainNode = context.createGainNode(0);
        mainNode.connect(context.destination);

        var clip = context.createBufferSource();

        context.decodeAudioData(xhr.response, function (buffer) {
            clip.buffer = buffer;
            clip.gain.value = 1.0;
            clip.connect(mainNode);
            clip.loop = true;
            clip.noteOn(0);
        }, function (data) {});
    }
    catch(e) {
        console.warn('Web Audio API is not supported in this browser');
    }
};

