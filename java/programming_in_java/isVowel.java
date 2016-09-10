
public class Word
{
    private String letters;
    
    public Word(String letters) 
    {
        this.letters = letters.toLowerCase();
    }
    
    /** Is the ith letter a vowel? */
    public boolean isVowel(int i)
    {
        // TODO your code here!
        // A little piece of code to get you started
        return letters.substring(i, i+1).equals("a")
            || letters.substring(i, i+1).equals("e")
            || letters.substring(i, i+1).equals("i")
            || letters.substring(i, i+1).equals("o")
            || letters.substring(i, i+1).equals("u");
    }
    
    /** Is the ith letter a consonant? */
    public boolean isConsonant(int i)
    {
        return !isVowel(i);
    }
}
