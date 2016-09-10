//Implement a class Balloon that models a spherical balloon that is being filled with air.  
//The constructor constructs an empty balloon (That is, the volume is 0).

//Supply these methods: 
//	void addAir(double amount) adds the given amount of air
//	double getVolume() gets the current volume
//	double getSurfaceArea() gets the current surface area
//	double getRadius() gets the current radius

//For the draft, supply the constructor and the getVolume method. Provide the instance variable.

//See this link for formulas for volume and surface area: 
//    http://math.about.com/od/formulas/ss/surfaceareavol.htm 

//Hints
//	You can find the nth root of a number by using the pow method from the Math class 
//  where the exponent is 1/n.
//	You will need to solve the volume formula for radius.


/**
 * models a spherical balloon that is being filled with air
 */
public class Balloon
{
    private double vol;
    public Balloon() { vol = 0.0; }
    void addAir(double amount) { vol += amount; }
    double getVolume() { return vol; }
    double getSurfaceArea() { double r = getRadius(); return 4.0 * Math.PI * r * r; }
    double getRadius(){ return Math.pow(vol/(4.0/3.0 * Math.PI), 1.0/3.0); }

}



/**
 * Tests the Balloon class
 * You don't need to modify or change this file
 */
public class BalloonTester
{
   public static void main(String[] args)
   {
       Balloon balloon = new Balloon();
       balloon.addAir(100);
       System.out.println(balloon.getRadius());
       System.out.println("expected: 2.8794119114848606");
       System.out.println(balloon.getVolume());
       System.out.println("expected: 100.0");
       System.out.println(balloon.getSurfaceArea());
       System.out.println("expected: 104.18794157356089");
       
       balloon.addAir(100);
       System.out.println(balloon.getRadius());
       System.out.println("expected: 3.6278316785978095");
       System.out.println(balloon.getSurfaceArea());
       System.out.println("expected: 165.38804805627188");
   }
}
