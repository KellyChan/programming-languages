# Read from the file words.txt and output the word frequency list to stdout.
# text reader | word parser | sort | word count | print with filter | result sort
cat ./tests/para4.txt | tr ' ' '\n' | sort -rk2 | uniq -c | awk '{ if ($2 != "") print $2 " " $1}' | sort -rk2 -n
