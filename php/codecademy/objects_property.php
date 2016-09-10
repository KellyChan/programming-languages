<!DOCTYPE html>
<html>
	<head>
	  <title>Reconstructing the Person Class</title>
      <link type='text/css' rel='stylesheet' href='style.css'/>
	</head>
	<body>
      <p>
        <!-- Your code here -->
        <?php
            class Person{
                public $isAlive = true;
                public $firstname;
                public $lastname;
                public $age;

            }
            
            $teacher = new Person();
            $student = new Person();
            
            echo $teacher->isAlive;
        ?>
      </p>
    </body>
</html>
