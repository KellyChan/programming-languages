
GUIDANCE
Add constraints to limit phone number values
max_number : 999_999_9999
min_number : 100_000_0000
constraint name : user choice

SYNTAX
keep CONSTRAINT_NAME is BOOLEAN_EXPRESSION;

<'
struct txt { 

   sender : 	uint;
   receiver :	uint;

   // constraints
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




