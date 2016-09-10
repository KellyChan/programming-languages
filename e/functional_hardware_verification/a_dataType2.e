
GUIDANCE
Problem: sender, receiver use duplicate constraints
Solution: introduce new user defined data type
Using ranges on type definition removes need for explicit constraints

new data type name: phone_number


SYNTAX: type
type NAME : TYPE [FROM..TO];


<'
// insert user defined type BEFORE struct here
type phone_number : uint(bits:34) [100_000_0000..999_999_9999];
struct txt {
 
   // replace type with new type defined above
   sender :     phone_number;
   receiver :   phone_number;

   message :    string;
  
   fill() is {
  	
      message  = "This is a test!";
  	
   };
	
};


'>


