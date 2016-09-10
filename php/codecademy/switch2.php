<!DOCTYPE html>
<html>
	<head>
		<title></title>
	</head>
	<body>
    <?php
    $i = 5;
    
    switch ($i):
        case 0:
            echo '$i is 0.';
            break;
        case 1: echo ""; break;
        case 2: echo ""; break;
        case 3: echo ""; break;
        case 4: echo ""; break;
        case 5:
            echo '$i is somewhere between 1 and 5.';
            break;
        case 6:
        case 7:
            echo '$i is either 6 or 7.';
            break;
        default:
            echo "I don't know how much \$i is.";
    endswitch;
    ?>
    </body>
</html>
