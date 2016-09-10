var xmlhttp;

function loadXMLDoc(url) {

    xmlhttp = null;

    if (window.XMLHttpRequest) {
        // code for Firefox, Opera, IE7, etc
        xmlhttp = new XMLHttpRequest();
    } else {
        // code for IE6, IE5
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }

    if (xmlhttp != null) {
        xmlhttp.onreadystatechange = state_Change;
        xmlhttp.open("GET", url, true);
        xmlhttp.send(null);
    } else {
        alert("Your browser does not support XMLHTTP.");
    }
}


function state_Change() {
    if (xmlhttp.readyState == 4) {
        // 4 = "loaded"
        if (xmlhttp.status == 200) {
            // 200 = "OK"
            document.getElementById("text").innerHTML = xmlhttp.responseText;
            document.getElementById("headers").innerHTML = xmlhttp.getAllResponseHeaders();
            document.getElementById("headers-modified").innerHTML = xmlhttp.getResponseHeader('Last-Modified');
        } else {
            alert("Problem retrieving data: " + xmlhttp.statusText);
        }
    }
}

