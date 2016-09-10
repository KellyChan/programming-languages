// Bluej project: twitter
//  TODO: Write code for the shorten(String longPost) method.
public class Twitterizer
{
    /**
     * Shortens and returns long posts by removing vowels
     * @param longPost the post whos vowels need to be removed
     * @return a version of the post without vowels
     */
    public String shorten(String longPost)
    {
        // YOUR CODE HERE
        String vowels = "aeiouAEIOU";
        for (int i = 0; i < vowels.length(); i ++) 
        {
            longPost = longPost.replace(vowels.substring(i, i + 1), "");
        }
        return longPost;
    }
}
