<html>
  <head>
    <title>Override!</title>
  </head>
  <body>
    <p>
      <?php
        class Vehicle {
          public function honk() {
            return "HONK HONK!";
          }
        }
        // Add your code below!
        class Bicycle extends Vehicle{
            public function honk(){
                return "Beep beep!";
            }
        }
        
        $bicycle = new Bicycle();
        echo $bicycle->honk();
      ?>
    </p>
  </body>
</html>
