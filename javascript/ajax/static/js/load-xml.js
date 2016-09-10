var xmlhttp;

function loadXMLDoc(url) {
    xmlhttp = null;

    if (window.XMLHTTPRequest) {
        // code for IE7, Firefox, Opera, etc
        xmlhttp = new XMLHttpRequest();
    } else if (window.ActiveXObject) {
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
    if (xmlhttp.readyState==4) {
        // 4 = "loaded"
        if (xmlhttp.status==200) {
            // 200 = "OK"
            document.getElementById('A1').innerHTML = xmlhttp.status;
            document.getElementById('A2').innerHTML = xmlhttp.statusText;
            document.getElementById('A3').innerHTML = xmlhttp.responseText;
        } else {
            alert("Problem retrieving XML data:" + xmlhttp:statusText);
        }
    }
}

