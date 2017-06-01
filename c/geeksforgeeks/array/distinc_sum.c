int findSum(int array[], int n)
{
  sort(array, array+n);

  int sum = 0;
  for (int i = 0; i < n; ++i)
  {
    if (array[i] != array[i+1])
      sum = sum + array[i];
  }
  return sum;
}


void findSum_2(int array[], int n)
{
  int sum = 0;
  unordered_set<int> s;
  for (int i = 0; i < n; ++i)
  {
    if (s.find(array[i]) == s.end())
    {
      sum += array[i];
      s.insert(array[i]);
    }
  }
  return sum;
}


int main()
{
  int array[] = {1, 2, 3, 1, 1, 4, 5, 6};
  int n = sizeof(array)/sizeof(int);
  
  findSum(array, n);  

  return 0;
}
