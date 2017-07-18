void scalar_add (int n, const float * a, const float * b, float * result)
{
  int i;
  for (i = 0; i < n; ++i)
    result[i] = a[i] + b[i];
}
