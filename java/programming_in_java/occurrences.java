// BlueJ Project: lesson4/book7
// Video: Working with the Book Text
    
import java.util.Scanner;

public class LookUpAnyWord
{
    public static void main(String[] args)
    {
        System.out.print("Type a word and I'll tell you how many times it appears: ");
        // TODO: Create an scanner object to read the user input
        // Read a word from the scanner and assign it to a String variable named word
        Scanner in = new Scanner(System.in);
        String word = in.nextLine();

        // TODO: Create a book object that reads from aliceInWonderland.txt
        Book aliceInWonderland = new Book("aliceInWonderland.txt");


        // TODO: Find the number of occurences of that word and assign it to a variable named occurences
        int occurrences = aliceInWonderland.occurrencesOf(word);
        System.out.println(word + " occurs " + occurrences + " times!");
    }
}
