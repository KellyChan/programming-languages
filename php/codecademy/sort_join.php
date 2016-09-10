<html>
    <p>
	<?php
	// Create an array with several elements in it,
	// then sort it and print the joined elements to the screen
	$names = array("david","alana","elizabeth");
	sort($names);
	print join(", ", $names);

	?>
	</p>
	<p>
	<?php
	// Reverse sort your array and print the joined elements to the screen
	rsort($names);
	print join(", ",$names);

	?>
	</p>
</html>
