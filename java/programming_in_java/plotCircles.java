// BlueJ Project: lesson4/fields

public class Plot
{
    private int width;
    private int length;

    /**
        Constructs a plot of a given size.
        @param width the width of the plot
        @param length the length of the plot
    */
    public Plot(int width, int length)
    {
        this.width = width;
        this.length = length;
    }

    /**
     * Calculates the number of circular fields that can fit on this plot.
     */
    public int getNumberOfFields(int radius)
    {
        // TODO: Using the formulas that we developed, compute
        // the total number of rows, then
        // the number of even and odd rows,
        // the number of columns in each even and odd row,
        // and finally the total number of fields
        int diameter = 2 * radius;
        int nRows = (int) ((length - diameter ) / (radius * Math.sqrt(3))) + 1;
        int oddRows = (nRows + 1) / 2;
        int evenRows = nRows / 2;
        int colsPerOddRow = width / diameter;
        int colsPerEvenRow = (width - radius) / diameter;
        int totalFields = oddRows * colsPerOddRow + evenRows * colsPerEvenRow;
        return totalFields;
    }
}



public class PlotTester
{
    public static void main(String[] args)
    {
        Plot plot = new Plot(1900, 4000);
        System.out.println(plot.getNumberOfFields(400));
        System.out.println("Expected: 8");
        System.out.println(plot.getNumberOfFields(300));
        System.out.println("Expected: 18");
    }
}
