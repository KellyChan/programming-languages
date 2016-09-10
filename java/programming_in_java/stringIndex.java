import java.util.Scanner;
import java.io.File;

public class Book
{
    private String bookText;

    public Book(String fileName)
    {
        readBook(fileName);
    }

    /**
     * Calculates the number of characters (as in letters and
     * symbols, not people) in the book.
     * @return the number of letters and symbols in the book.
     */
    public int getNumCharacters()
    {
        return bookText.length();
    }

    /**
       Finds where the string "Mad Hatter" occurs for the first time in the book.
       @return the location of the first occurrence of the string
    */
    
    public int firstOccurrenceOfMadHatter()
    {
        // TODO: Complete this method
        //
        // Check the fact sheet. There is a String function
        // that can do this for you.
        return bookText.indexOf("Mad Hatter");
    }

    /**
     * A method to help read the book out of the file.
     * You don't have to read this unless you want to.
     * the "try" and "catch" are java's way of dealing with
     * runtime errors or "exceptions".
     */
    public void readBook(String fileName)
    {
        bookText = "";
        try
        {
            Scanner file = new Scanner(new File(fileName));
            while (file.hasNextLine())
            {
                String line = file.nextLine();
                bookText += line + "\n";
            }
            file.close();
        }
        catch (Exception e)
        {
            System.out.println(e);
        }
    }
}


// Mad Hatter search
public class BookTester
{
    public static void main(String[] args)
    {
        Book alice = new Book("aliceInWonderland.txt");

        System.out.println(alice.firstOccurrenceOfMadHatter());
        System.out.println("Expected: -1");

        Book code = new Book("BookTester.java");
        System.out.println(code.firstOccurrenceOfMadHatter());
        System.out.println("Expected: 3");
    }
}
