// BlueJ project: lesson7/gallery2

import java.util.ArrayList;

public class ListOfPictures
{
    public static void main(String[] args)
    {
        ArrayList<Picture> gallery = new ArrayList<Picture>();
        gallery.add(new Picture("degas1.jpg"));
        gallery.add(new Picture("gaugin1.jpg"));
        gallery.add(new Picture("monet1.jpg"));
        gallery.add(new Picture("monet2.jpg"));
        gallery.add(new Picture("renoir1.jpg"));

        for(int i = 1; i < gallery.size(); i++)
        {
            Picture pic = gallery.get(i);
            Picture left = gallery.get(i - 1);
            // Move pic 10 pixels to the right of left
            // Hint: left.getMaxX()
            pic.translate(left.getMaxX() + 10, 0);
            pic.draw();
            
        }
        
        for (int i = 0; i < gallery.size(); i++)
        {
            Picture pic = gallery.get(i);
            pic.draw();
        }
    }
}
