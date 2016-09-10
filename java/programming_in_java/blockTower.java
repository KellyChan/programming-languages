// Name the class BlockTower. Each block has a width of 40 and a height of 30. 
// There are three rows. The upper left-hand corner of the bottom row is at (20, 70)
 
// Create a custom color for the topmost color where red = 125, green=125, blue = 255. 
// For all the other colors, use predefined colors. 
// The colors from top to bottom are: custom, RED, PINK, BLUE, MAGENTA, CYAN, GREEN,
// YELLOW, ORANGE, and GREEN
 
// For the draft, draw the first rectangle in the bottom row like the image below:
// http://imgur.com/0LPUjzB
 
// HINT:
// The rectangle should have the attributes:
// x: 20
// y: 70
// width: 40
// height: 30
//
// You should set the color of the rectangle to blue by using the method:
// setColor(Color.somecolor); on the rectangle object that you created
//
// Don't forget to draw the rectangle by calling the fill() function
 
public class BlockTower
{
   public static void main(String[] args)
   {
      // your code here
       Rectangle block11 = new Rectangle(20,70,40,30);
       Rectangle block12 = new Rectangle(60,70,40,30);
       Rectangle block13 = new Rectangle(100,70,40,30);
       Rectangle block21 = new Rectangle(40,40,40,30);
       Rectangle block22 = new Rectangle(80,40,40,30);
       Rectangle block31 = new Rectangle(60,10,40,30);
      
     
       Color custom = new Color(125,125,255);
       block31.setColor(custom);
       block21.setColor(Color.RED);
       block22.setColor(Color.PINK);
       block11.setColor(Color.BLUE);
       block12.setColor(Color.MAGENTA);
       block13.setColor(Color.CYAN);
                        
       block31.fill();
       block21.fill();
       block22.fill();
       block11.fill();
       block12.fill();
       block13.fill();
                       
 
   }
}