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
    var folder_id = response.data.folders[0].id;
    WL.upload({
        path: folder_id,
        element: 'file-to-save',
        overwrite: 'rename'
    }).then(
        function(response) {
            // Handle the response
            $('#status').html("Upload complete. Getting shared link...");
            // TODO: Get the file id
            // var file_id = ...
            var file_id = response.id;
            getSharedLink(file_id);
        },
        function(error) {
            // Handle errors
            $('#status').html(error.error.message);
        }
    );
}

// WL.ui calls this if there was an error in selecting
// a save location on SkyDrive, or if the user canceled
function handleError(errorResponse) {
    console.log("Error saving to SkyDrive");
}

// Complete this function, which generates a read-only shared link
// Given a file_id. Call showResult on the link you generate.
function getSharedLink(file_id) {
    // TODO: add a WL.api call here to get a shared link using the file_id
    WL.api({
    path: file_id+"/shared_read_link",
    method: "GET"
}).then(
    function(response) {
        // Get the link and call showResult on it
        var link = response.link;
        showResult(link);
    },
    function(responseFailed) {
        // Handle an error
        console.log(responseFailed.error.message);
    }
);
}

// Show the fruits of your labor and submit answer for evaluation
// Don't edit this!
function showResult(shared_link) {
    // Display the link
    $('#share-link').val(shared_link);
    $('#share-link').trigger('c');
}
