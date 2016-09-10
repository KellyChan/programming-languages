// BlueJ Project: lesson4/book3
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
        // TODO: Return the first sentence
        int endOfSentence = bookText.indexOf(".") + 1;
        return bookText.substring(0,endOfSentence);
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
        System.out.println(alice.getFirstSentence());
        System.out.println("Expected: Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'.");

        Book mary = new Book("mary.txt");

        // Here, the end-of-line markers are replaced with spaces so that the actual and expected results fit on a line.
        System.out.println(mary.getFirstSentence().replace("\n", " "));
        System.out.println("Expected: Mary had a little lamb, little lamb, little lamb, Mary had a little lamb, its fleece was white as snow.");
    }
}
