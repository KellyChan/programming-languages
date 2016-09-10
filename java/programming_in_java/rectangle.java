// You need to construct and draw two rectangles for this quiz.
//
// Both rectangles have width 20 and height 30. The first rectangle
// has top left corner at (60, 90). The bottom right corner of the 
// first rectangle should be the top left corner of the second 
// rectangle. 
public class RectangleDemo
{
    public static void main(String[] args)
    {
       // TODO: your code here
        Rectangle box1 = new Rectangle(60, 90, 20, 30); // arguments(x,y,width,heigh)
        Rectangle box2 = new Rectangle(80, 120, 20, 30);
        box1.draw();
        box2.draw();
    }
}