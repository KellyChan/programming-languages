<html>
  <head>
    <title>If, Elseif, and Else</title>
  </head>
  <body>
    <p>
      <?php
        $guess = 7;
        $number = 7;
        
        // Write your if/elseif/else statement here!
        if ($guess < $number){
            echo "Too low!";
        }elseif($guess > $number){
            echo "Too high!";
        }else{
            echo "You win!";
        }
        
      ?>
    </p>
  </body>
</html>
