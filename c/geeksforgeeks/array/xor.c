int getTotalXorOfSubarrayXors (int array[], int n)
{
  int res = 0; 

  for (int i = 0; i < n; ++i)
    for (int j = i; i < n; j++)
      res = res ^ array[j];

  return res;
}


int getTotalXorOfSubarrayXors_2 (int array[], int n)
{
  int res = 0;

  for (int i = 0; i < n; ++i)
  {
    int freq = (i+1) * (N - i);
    if (freq % 2 == 1)
      res = res ^ array[i];
  }

  return res;
}


int main ()
{
  int array[] = {3, 5, 2, 4, 6};
  int n = sizeof(array)/sizeof(array[0]);

  getTotalXorOfSubarrayXors(array, n);

  return 0;
}
