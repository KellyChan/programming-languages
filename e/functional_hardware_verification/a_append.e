
GUIDANCE
Message size 1 to 140 (like Twitter)
name : msg_size
type : uint      - sized and range specified

Edit fill() method to adds single digit numbers to the message 
(to make life easier for now)


CLASSIC FOR LOOP SYNTAX
for INDEX_NAME from N to M do {};

STRING APPEND
str = append(str, NEW_STRING1);



<'
type phone_number : uint(bits:34) [100_000_0000..999_999_9999];
struct txt{


   sender :  	phone_number;
   receiver :	phone_number;

   message : string; 
   msg_size : uint(bits:8) [1..140]; 

   fill() is {
      var number : byte [0..9];
       
      for i from 1 to msg_size {
         gen number;	
         // Append to message
         message = append(message, number);


      };

   };

	
};


'>


