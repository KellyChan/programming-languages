//A landscape designer needs a program to make calculations in meters about a front yard 
//given its width and length in feet.  Write a class YardCalculator with a constructor that takes 
//the width and length in feet as arguments
//public YardCalculator(double length, double width)

//Provide methods :

//public double perimeter() which returns the perimeter in meters
//public double area() which returns the area in square meters
//public double diagonal() which returns the diagonal in meters

//1 foot = 0.3048 meters.

//Define and use a constant for the conversion factor. Define it in the class itself, 
//outside any method, so all methods in any class can see it.

//      public static final double METERS_PER_FOOT = 0.3048;

//HINT: diagonal = square root of (width^2 plus height^2)

//For the draft, provide the instance variables and  the constructor


public class YardCalculator
{
    
    public static final double METERS_PER_FOOT = 0.3048;
    double length;
    double width;

   /**
    * Constructs a YardCalculator with the given parameters
    * @param theLength the length of the yard
    * @param theWidth the width of the yard
    */
   public YardCalculator(double theLength, double theWidth)
   {
       this.length = theLength;
       this.width = theWidth;
   }

   /**
    * Gets the perimeter of the yard
    * @return the perimeter
    */
   public double perimeter()
   {
       return 2* (width + length) * METERS_PER_FOOT;

   }

   /**
    * Gets the area of the yard
    * @return the area
    */
   public double area()
   {
       return width * length * Math.pow(METERS_PER_FOOT,2);

   }

    /**
    * Gets the diagonal of the yard
    * @return the diagonal
    */
   public double diagonal()
   {
       
       return Math.sqrt(Math.pow(width * METERS_PER_FOOT,2) + Math.pow(length * METERS_PER_FOOT,2));

   }

}


// You don't need to modify this file

public class YardCalculatorTester
{
    public static void main(String[] args)
    {
        double length = 40.0;
        double width = 30.0;

        YardCalculator calculator = new YardCalculator(length, width); 
        
        double perimeter = calculator.perimeter();
        double area = calculator.area();
        double diagonal = calculator.diagonal();

        System.out.println("perimeter: " + perimeter);
        System.out.println("Expected: 42.672");
        System.out.println("diagonal: " + diagonal);
        System.out.println("Expected: 15.24");

        length = 60.1;
        width = 35.5;
        
        calculator = new YardCalculator(length, width); 
        perimeter = calculator.perimeter();
        area = calculator.area();
        diagonal = calculator.diagonal();

        System.out.printf("perimeter: %.2f\n", perimeter);
        System.out.println("Expected: 58.28");
        System.out.printf("area: %.2f\n", area);
        System.out.println("Expected: 198.21");
        System.out.printf("diagonal: %.2f\n", diagonal);
        System.out.println("Expected: 21.28");        
    }
}
