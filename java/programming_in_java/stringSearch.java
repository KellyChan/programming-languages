// BlueJ Project: lesson4/book4
// Video: Working with the Book Text

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
       Gets the first sentence of this book.
       @return the first sentence (including the period) of this book.
    */
    public String getFirstSentence()
    {
        int endOfSentence = bookText.indexOf(".");
        return bookText.substring(0, endOfSentence + 1);
    }

    /**
       Gets the second sentence of this book.
       @return the second sentence (including the period) of this book.
    */
    public String getSecondSentence()
    {
        // TODO: Get the second sentence
        // Locate the period of the first sentence
        // Return the substring consisting of everything after that period up to
        // the second period
        int beginningOfSentence = bookText.indexOf(".") + 1;
        int endOfSentence = bookText.indexOf(".", beginningOfSentence);
        return bookText.substring(beginningOfSentence, endOfSentence + 1);
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


public class BookTester
{
    public static void main(String[] args)
    {
        Book alice = new Book("chapter1.txt");

        // The sentence is pretty long, but it is in a single line
        System.out.println(alice.getSecondSentence());
        System.out.println("Expected: So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.");

        Book mary = new Book("mary.txt");

        // Here, the end-of-line markers are replaced with a space so that the actual and expected results fit on a line.
        System.out.println(mary.getSecondSentence().replace("\n", " "));
        System.out.println("Expected: And everywhere that Mary went, Mary went, Mary went, and everywhere that Mary went, the lamb was sure to go.");
    }
}
