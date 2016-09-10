<html>
    <p>
        <?php
            $name = "Trey";
            // Print out the position of a letter that is in
            // your own name
            $needle = "r44e";
            $pos = strpos($name,$needle);

            if ($pos == false) {
                print "$pos Sorry no $needle in $name";
            } else {
                print $pos . "Yes $needle is in $name.";
            }
        ?>
    </p>
    <p>
        <?php
            // Check for a false value of a letter that is not
            // in your own name and print out an error message
            $needle = "re";
            $pos = strpos($name,$needle);

            if ($pos == false) {
                print $pos . "Sorry no $needle in $name";
            } else {
                print $pos . "Yes $needle is in $name.";
            }
        ?>
    </p>
</html>
