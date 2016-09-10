// BlueJ project: lesson3/cars5
 
// Complete the second constructor of this class.
 
public class Car
{
    private double milesDriven;
    private double gasInTank;
    private double milesPerGallon;
    private Picture pic;
 
    public Car(double mpg)
    {
        milesDriven = 0;
        gasInTank = 0;
        milesPerGallon = mpg;
        pic = new Picture("car.jpg");
        pic.draw();
    }
 
    public Car(double mpg, String pictureFile)
    {
        // TODO: Complete this constructor
        milesDriven = 0;
        gasInTank = 0;
        milesPerGallon = mpg;
        pic = new Picture(pictureFile);
        pic.draw();
    }
 
    public void drive(double distance)
    {
        milesDriven = milesDriven + distance;
        double gasConsumed = distance / milesPerGallon;
        gasInTank = gasInTank - gasConsumed;
        int pixelsPerMile = 10;
        pic.translate(distance * pixelsPerMile, 0);
    }
 
    public void addGas(double amount)
    {
        gasInTank = gasInTank + amount;
    }
 
    public double getMilesDriven()
    {
        return milesDriven;
    }
 
    public double getGasInTank()
    {
        return gasInTank;
    }
}
 
 
// you don't need to modify or change this file
public class CarDemo
{
    public static void main(String[] args)
    {
        Car car1 = new Car(10, "hummer.jpg");
        Car car2 = new Car(50, "smart.jpg");
        car1.addGas(20);
        car1.drive(20);        
        car2.addGas(20);
        car2.drive(80);
        new Text(200, 120, "Gas left: " + car1.getGasInTank()).draw();
        new Text(800, 120, "Gas left: " + car2.getGasInTank()).draw();
    }
}