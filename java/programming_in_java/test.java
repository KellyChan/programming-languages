// BlueJ project lesson2/growtester
// 
// Print out the actual AND expected values for the
// x- and y-position, the width, and the height of the 
// rectangle after growing it. 
// To print out a value and its expected value, use 
// use this format: 
 
// System.out.println(box.getX());
// System.out.println("Expected: 50");
// System.out.println(box.getY());
// System.out.println("Expected: 50");
// ...
 
// Replace 50 with the value that you actually expect.
 
public class RectangleDemo
{
	public static void main(String args[])
	{
		Rectangle box = new Rectangle(45, 90, 60, 90);
		box.grow(20, 20);
        
        // You need to print out the following values of box and its expected values:
        //
        // X, by using box.getX();
        // Y, by using box.getY();
        // width, by using box.getWidth();
        // height, by using box.getHeight();
        // 
        // To print out a value and its expected value, use System.out.println(...);
        // For example:
        // System.out.println(box.getY());
        // System.out.println("Expected: " + 50);
        // 
		// Print out the actual and test values below this line:
        System.out.println(box.getX());
        System.out.println("Expected: 25"); //X
        System.out.println(box.getY());
        System.out.println("Expected: 70"); //Y
        System.out.println(box.getWidth());
        System.out.println("Expected: 100"); //Width
        System.out.println(box.getHeight());
        System.out.println("Expected: 130"); //Height
        
	}
}