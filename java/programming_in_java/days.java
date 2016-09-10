public class DaysAlivePrinter
{
	public static void main(String[] args)
	{
		// Sally Ride was born on May 26 1951
        // Her last day was July 23 2012
        // Use Day objects to calculate and print
        // how many days she lived. 
        // You'll want to use the daysFrom method.
        // Do you expect to get a negative or a positive number of days?
        Day birthday = new Day(1951, 5, 26);
        Day lastDay = new Day(2012, 5, 26);
        int daysAlive = lastDay.daysFrom(birthday);
        System.out.println(daysAlive);
	}
}