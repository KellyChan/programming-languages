STEPS
1 Start with pre-populated code
2 Guidance on exercise
3 Implement/edit code
4 (optional) Run code to assess results
5 Submit code for automatic assessment
6 Solution discussion

----------------------------------------
GUIDANCE
Field name: sender, receiver
Type: uint
Value: 212_555_0123, 408_555_0199

Field name: message
Type: string
Value: "This is a test!"
----------------------------------------

----------------------------------------
SYNTAX: field declaration
(statement, can only be done in structs)

FIELD_NAME : TYPE;
----------------------------------------

----------------------------------------
SYNTAX: assignment 
(action, can only be done in methods)

FIELD_NAME = VALUE;
----------------------------------------


<'
struct txt {

    // fields of data model for txt
    sender : uint;
    receiver : uint;
    
    message : string;
 
    // this is a method with no parameters
    fill() is {
        // assignments
        sender = 212_555_0123;
        receiver = 408_555_0199;
        
        message = "This is a test!";

    };

};
'>

