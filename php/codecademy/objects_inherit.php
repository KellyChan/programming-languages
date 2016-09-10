<html>
  <head>
    <title>The Shape of Things to Come</title>
  </head>
  <body>
    <p>
      <?php
        class Shape {
          public $hasSides = true;
        }
        
        class Square extends Shape {
        
        }
        
        $square = new Square();
        // Add your code below!
        if (property_exists($square, "hasSides")) {
          echo "I have sides!";
        }
      ?>
    </p>
  </body>
</html>
