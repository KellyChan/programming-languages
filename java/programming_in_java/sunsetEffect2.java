// BlueJ project: lesson4/sunsetInput
// Video: Color Your Own Sunset Effect
 
public class Color
{
    private int red;
    private int green;
    private int blue;
 
    /**
     * Makes this color redder.
     */
    public void redden(int addedRed) // TODO: Add a parameter
    {
        final int MAX_RED = 255;
        red = Math.min(MAX_RED, red + addedRed);
    }
 
    /**
     * Gets the red value of this color.
     * @return the red value (between 0 and 255)
     */
    public int getRed()
    {
        return red;
    }
 
    /**
     * Gets the green value of this color.
     * @return the green value (between 0 and 255)
     */
    public int getGreen()
    {
        return green;
    }
 
    /**
     * Gets the blue value of this color.
     * @return the blue value (between 0 and 255)
     */
    public int getBlue()
    {
        return blue;
    }
 
    /**
     * Constructs a new Color object.
     * @param red the red value of the color (between 0 and 255)
     * @param green the green value of the color (between 0 and 255)
     * @param blue the blue value of the color (between 0 and 255)
     */
    public Color(int red, int green, int blue)
    {
        this.red = red;
        this.green = green;
        this.blue = blue;
    }
 
    // Color constants
 
    public static final Color RED = new Color(255, 0, 0);
    public static final Color GREEN = new Color(0, 255, 0);
    public static final Color BLUE = new Color(0, 0, 255);
    public static final Color WHITE = new Color(255, 255, 255);
    public static final Color LIGHT_GRAY = new Color(192, 192, 192);
    public static final Color GRAY = new Color(128, 128, 128);
    public static final Color DARK_GRAY = new Color(64, 64, 64);
    public static final Color BLACK = new Color(0, 0, 0);
    public static final Color CYAN = new Color(0, 255, 255);
    public static final Color MAGENTA = new Color(255, 0, 255);
    public static final Color YELLOW = new Color(255, 255, 0);
    public static final Color PINK = new Color(255, 175, 175);
    public static final Color ORANGE = new Color(255, 200, 0);
}
 
 
// BlueJ project: lesson4/sunsetInput
// Video: Color Your Own Sunset Effect
 
// TODO: Add import statement
 
 
import java.util.Scanner;
 
public class Sunset
{
    public static void main(String[] args)
    {
        Picture pic = new Picture();
        pic.load("queen-mary.png");
 
        System.out.print("Added redness: ");
        // TODO: Read added redness from scanner
        Scanner in = new Scanner(System.in);
        //System.out.print("Added redness: ");
        int addedRed = in.nextInt();
 
        pic.draw();
        for (int i = 0; i < pic.pixels(); i++)
        {
            Color c = pic.getColorAt(i);
            c.redden(addedRed); // TODO: Pass added redness to method
            pic.setColorAt(i, c);
        }
    }
}