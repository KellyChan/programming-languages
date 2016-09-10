<html>
    <p>
	<?php
	// Create an array and push on the names
    // of your closest family and friends
    $names = array("david","terry","elizabeth");
    array_push($names, "barbara");
    
	// Sort the list
	sort($names);

	// Randomly select a winner!
	$winner = $names[rand(0,count($names))];

	// Print the winner's name in ALL CAPS
	print strtoupper($winner);
	?>
	</p>
</html>
