
var setup = function() {

// Create a new XMLHttpRequest called 'soundRequest',
// that GETs the file '/media/audio/gamedev/bg_menu.ogg'.
//
// To properly read this binary file, we'll need
// to specify the responseType of the request as
// an 'arraybuffer'.
//
// Doing this is necessary to work with any kind
// of binary data, like sound files, rather than
// text data.
//
// WARNING: If you don't specify a responseType
// of 'arraybuffer', your browser will try to
// interpret the sound file as text data. This
// could cause your browser to slow to a crawl or
// worse.
//
// Once you have done this, leave the request's
// onload to the below function. This will play
// the sound that you loaded.
//
// Don't worry if you don't understand what this
// code does, we'll be going over it later!
//
// YOUR CODE HERE
    
        var soundRequest = new XMLHttpRequest();
    
        soundRequest.open('GET','/media/audio/gamedev/bg_menu.ogg',true);
        soundRequest.responseType = 'arraybuffer';

	soundRequest.onload = function () {

		try {
			var context = new webkitAudioContext();

			var mainNode = context.createGainNode(0);
			mainNode.connect(context.destination);

			var clip = context.createBufferSource();

			context.decodeAudioData(soundRequest.response, function (buffer) {
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

	soundRequest.send();
};

