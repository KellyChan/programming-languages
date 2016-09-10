var xmlhttp;

function loadXMLDoc(url) {
    xmlhttp = null;

    if (window.XMLHttpRequest) {
        // code for IE7, Firefox, Mozilla, etc
        xmlhttp = new XMLHttpRequest();
    } else if (window.ActiveXObject) {
        // code for IE5, IE6
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }

    if (xmlhttp != null) {
        xmlhttp.onreadystatechange = onResponse;
        xmlhttp.open("GET", url, true);
        xmlhttp.send(null);
    } else {
        alert("Your browser does not support XMLHTTP.");
    }
}


function onResponse() {

    if (xmlhttp.readyState != 4) return;

    if (xmlhttp.status != 200) {
        alert("Problem retrieving XML data");
        return;
    }

    txt = "<table border='1'>";
    x = xmlhttp.responseXML.documentElement.getElementsByTagName("CD");
    for (i=0; i< x.length;i++) {
        txt = txt + "<tr>";
        xx = x[i].getElementsByTagName("TITLE");

        {
            try {
                txt = txt + "<td>" + xx[0].firstChild.nodeValue + "</td>";
            }

            catch (er) {
                txt = txt + "<td> </td>";
            }

        }

        xx = x[i].getElementsByTagName("ARTIST");

        {
            try {
                txt = txt + "<td>" + xx[0].firstChild.nodeValue + "</td>";
            }
            catch (er) {
                txt = txt + "<td> </td>";
            }
        }

        txt = txt + "</tr>";
    }

    txt = txt + "</table>";

    document.getElementById('copy').innerHTML = txt;
}

