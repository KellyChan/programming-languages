kernel void scalar_add (global const float * a, global const float * b, global float * result)
{
  int id = get_global_id(0);
  result[i] = a[id] + b[id];
}
