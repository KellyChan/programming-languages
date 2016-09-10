// BlueJ project: lesson3/friends3
public class Person
{
    private String name;
    private String friends;
    
    // TODO #1: Fill in the rest of the arguments to the Person constructor
    // The Person constructor should take in 3 additional arguments:
    // 1) String pictureName
    // 2) int xCoord
    // 3) int yCoord
    public Person (String aName, String pictureName, int xCoord, int yCoord)
    {
        name = aName;
        friends = "";
        
        // TODO #2: Fill in the rest of this constructor
        //
        // You need to create a local variable called picture and assign to
        // picture the picture that we want to use.
        //
        // You need need to move the picture to xCoord and yCoord.
        // You can move the picture object by calling the translate method.
        //
        // The last step is to draw the picture.
        Picture picture = new Picture(pictureName);
        picture.translate(xCoord, yCoord);
        picture.draw();
    }
    
    public void addFriend(Person friend) 
    {
        friends = friends + friend.name + " ";
    }
    
    public void unfriend(Person nonFriend)
    {
        friends = friends.replace(nonFriend.name + " ", "");
    }
    
    public String getFriends() 
    {
        return friends;
    }
}
 
 
// This is for your reference. You do not need to edit this file.
 
public class PersonDemo 
{
    public static void main(String[] args)
    {
        Person sara = new Person("Sara", "sara.jpg", 10, 200);
        Person chenghan = new Person("Cheng-Han", "cheng-han.png", 300, 0);
        Person cay = new Person("Cay", "cay.gif", 250, 180);
    }
}