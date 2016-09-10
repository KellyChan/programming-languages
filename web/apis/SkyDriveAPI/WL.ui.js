// The callback URL to redirect to after authentication
redirectUri = "http://external.codecademy.com/skydrive.html";

// Initialize the JavaScript SDK
WL.init({ 
    client_id: '000000004C0E2C11', 
    redirect_uri: redirectUri
});

$(document).ready(function() {
   // TODO: Add WL.ui here.
   WL.ui({
       name: "skydrivepicker",
       element: "skydrive-upload",
       mode: "save",
       onselected: handleUpload,
       onerror: handleError
});
});

// Have WL.ui call this once the user has successfully
// selected a save location on SkyDrive
function handleUpload(response) {
    // We'll write this in the next exercise
    showResult(response); // Don't change this or you won't pass
}

// Have WL.ui call this if there was an error in selecting
// a save location on SkyDrive, or if the user canceled
function handleError(responseFailure) {
    $('#result').html(responseFailure.error.message);
}

// Show the fruits of your labor and submit answer for evaluation
// Don't edit this!
function showResult(response) {
    $('#result').html("<h3>Roger! Got the folder you selected.</h3>");
    answer = response;
    $('#result').trigger('c');
}

var answer; // This is for Codecademy to check your results
