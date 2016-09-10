// The callback URL to redirect to after authentication
redirectUri = "http://external.codecademy.com/skydrive.html";

// Initialize the JavaScript SDK
WL.init({ 
    client_id: '000000004C0E2C11', 
    redirect_uri: redirectUri
});

$(document).ready(function() {
    //TODO: add WL.ui here
    WL.ui({
    name: "skydrivepicker",
    element: "skydrive-download",
    mode: "open",
    select: "multi",
    onselected: handleDownload,
    onerror: handleError
});
});

// WL.ui calls this once the user has successfully
// selected file(s) on SkyDrive
function handleDownload(response) {
    //TODO: Add WL.download here
    var files = response.data.files;
for (var i = 0; i < files.length; i++) {
    var file = files[i];
    WL.download({ 
        path: file.id + "/content" 
    });
}
    showResult(response);
}

// WL.ui calls this if there was an error in selecting
// a file(s) on SkyDrive, or if the user canceled
function handleError(failureResponse) {
    $('#result').html(failureResponse.error.message);
}

// Show the fruits of your labor and submit answer for evaluation
// Don't edit this!
function showResult(response) {
    $('#result').html("<h3>Woot! You should see your files downloading.</h3>");
    answer = response;
    $('#result').trigger('c');
}

var answer; // This is for Codecademy to check your results
