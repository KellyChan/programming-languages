//Complete the class you started in the draft for this problem. Here are the instuctions again:
 
// Create a class CreditCard that represents a credit card account.
// You need to construct a new credit card with a 0.0 balance. 
 
// The constructor will have this header:
// public CreditCard()
//
//CreditCard will have these methods:
//public void purchase(double amount)
//public void payment(double amount)
//public double getBalance()
 
// For the draft, implement the constructor and the instance variable, balance, which is a double.
 
/**
   A credit card has a balance that can be changed by
   purchases and payments.
*/
public class CreditCard
{
   private double balance;
   /**
      Constructs a credit card with a zero balance.
   */
   public CreditCard()
   {
     balance = 0.0;
   }
 
 
   /**
      Makes a payment to  credit card.
      @param amount the amount to pay
   */
   public void payment(double amount)
   {
       // TODO #1: You should subtract amount from balance variable in this function
       balance = balance - amount;
   }
 
   /**
      Makes a purchase with the credit card
      @param amount the amount of the purchase
   */
   public void purchase(double amount)
   {
       // TODO #2: You should add amount to the balance variable in this function
       balance = balance + amount;
   }
 
   /**
      Gets the current balance of the credit card.
      @return the current balance
   */
   public double getBalance()
   {
       // TODO #3: You should return the instance variable balance in this function
       return balance;
   }
}
 
 
// You don't need to modify this class
public class CreditCardTester
{
   public static void main(String[] args)
   {
       CreditCard visa =new CreditCard();
       System.out.println(visa.getBalance());
       System.out.println("Expected: 0.0");
       
       visa.purchase(10.50);
       visa.purchase(2.99);
       System.out.println(visa.getBalance());
       System.out.println("Expected: 13.49");
       
       visa.payment(5.0);
       visa.payment(5);
       System.out.println(visa.getBalance());
       System.out.println("Expected: 3.49");
       
   }
}