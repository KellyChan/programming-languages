/**
   A simulated car that consumes gas as it drives.
*/
public class Car
{
   private double milesDriven;
   private double gasInTank;
   private double milesPerGallon;
   
 
   /**
      Constructs a car with a given fuel efficiency.
   */
   public Car(double mpg)
   {
      milesDriven = 0;
      gasInTank = 0;
      milesPerGallon = mpg;
   }
 
  
   /**
       Adds gas to the tank of this car.
       @param amount the gas added
    */
   
   public void addGas(double amount)
   {
      gasInTank = gasInTank + amount;
   }
 
   /**
      Gets the current amount of gas in the tank of this car.
      @return the current gas level
   */
   public double getGasInTank()
   {
      return gasInTank;
   }
 
   /**
      Drives this car by a given distance.
      @param distance the distance to drive
   */
   public void drive(double distance)
   {
      this.milesDriven = this.milesDriven + distance;
      double gasConsumed = distance / this.milesPerGallon;
      this.gasInTank = this.gasInTank - gasConsumed;
   }  
 
   /**
      Gets the current mileage of this car.
      @return the total number of miles driven
   */
   public double getMilesDriven()
   {
      return milesDriven;
   }
}