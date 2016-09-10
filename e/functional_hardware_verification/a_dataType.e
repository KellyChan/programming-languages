
GUIDANCE
Problem: 32 bit integer too small to model 10 digit phone numbers 
Solution: increase size of data type


<'
struct txt { 

   // increase size of data type
   sender :    uint(bits:34);
   receiver :  uint(bits:34);
  
  
   keep max_sender is sender <= 999_999_9999;
   keep min_sender is sender >= 100_000_0000;
   keep max_receiver is receiver <= 999_999_9999;
   keep min_receiver is receiver >= 100_000_0000;
  

   message :     string;


   fill() is {

      message  = "This is a test!";

   };

};
'>




