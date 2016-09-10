// The callback URL to redirect to after authentication
redirectUri = "http://external.codecademy.com/skydrive.html";

// Initialize the JavaScript SDK
WL.init({ 
    client_id: '000000004C0E2C11', 
    redirect_uri: redirectUri
});

$(document).ready(function() {
    // Start the login process when the login button is clicked
    $('#login').click(function() {
        WL.login({
            //scope: ["scope1 scope2"]
            scope: ["wl.skydrive wl.signin"]
        }).then(
            // Handle a successful login
            function(response) {
                $('#status').html("<strong>Success! Logged in.</strong>");
            },
            // Handle a failed login
            function(responseFailed) {
                // The user might have clicked cancel, etc.
                $('#status').html(responseFailed.error.message);
            }
        );
    });
});

