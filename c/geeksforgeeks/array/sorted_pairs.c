int findPairs(int array[], int n, int x)
{
  int l = 0;
  int r = n - 1;
  int result = 0;

  while (l < r)
  {
    if (array[l] + array[r] < x)
    {
      result += (r - 1);
      l++;
    }
    else 
      r--;
  }
  return result;
}


int main ()
{
  int array[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int n = sizeof(array)/sizeof(int);

  int x = 7;
  findPairs(array, n, x);

  return 0;
}
