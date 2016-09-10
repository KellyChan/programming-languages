// Bluej project: lesson5/sierraFlag

public class Flag
{
    private Picture pic;
    private int width;
    private int height;


    public Color getColorAt(int x, int y)
    {
        Color c;
        if (x >= width/3 
            && x < width * 2/3 
            && y >= height / 3 
            && y < height * 2/3)
        {
            c = Color.BLUE;
        }
        else
        {
            c = Color.WHITE;
        }
        
        return c;
    }

    public Flag(int width, int height)
    {
        this.width = width;
        this.height = height;
        pic = new Picture(width, height);
        pic.draw();
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Color c = getColorAt(x, y);
                pic.setColorAt(x, y, c);
            }
        }
    }
}
