// The callback URL to redirect to after authentication
redirectUri = "http://external.codecademy.com/skydrive.html";

// Initialize the JavaScript SDK
WL.init({ 
    client_id: '000000004C0E2C11', 
    redirect_uri: redirectUri
});

$(document).ready(function() {
    WL.ui({
        name: "skydrivepicker",
        element: "skydrive-upload",
        mode: "save",
        onselected: handleUpload,
        onerror: handleError
    });
});

// WL.ui calls this once the user has successfully
// selected a save location on SkyDrive
function handleUpload(response) {
    //TODO: Add WL.upload here. Don't forget to call showResult() on its response.
    var folder = response.data.folders[0];
WL.upload({
    path: folder.id,
    element: 'file-to-save',
    overwrite: 'rename'
}).then(
    function(response) {
        // Handle the response
        showResult(response);
    },
    function(error) {
        // Handle errors
        $('#error').html(error.error.message);
    },
    function(progress) {
        // Handle progress events
    }
);
    
}

// WL.ui calls this if there was an error in selecting
// a save location on SkyDrive, or if the user canceled
function handleError(failureResponse) {
    $('#result').html(failureResponse.error.message);
}

// Show the fruits of your labor and submit answer for evaluation
// Don't edit this!
function showResult(response) {
    $('#result').html("<h4>Success! Your file was uploaded.</h4>");
    answer = response;
    $('#result').trigger('c');
}

var answer; // This is for Codecademy to check your results
