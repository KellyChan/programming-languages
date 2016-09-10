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
                
                public function __construct($firstname, $lastname, $age){
                    $this->firstname = $firstname;
                    $this->lastname = $lastname;
                    $this->age = $age;
                }
                
                public function greet(){
                    return "Hello, my name is " . $this->firstname . " " . $this->lastname . ". Nice to meet you! :-)";
                }

            }
            
            $teacher = new Person("boring","12345",12345);
            $student = new Person("me","me",2);
            
            echo $teacher->isAlive;
            echo $student->age;
            echo $teacher->greet();
            echo $student->greet();
            
        ?>
      </p>
    </body>
</html>
