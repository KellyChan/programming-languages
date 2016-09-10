def quicksort(xs: Array[Int]) {

  def swap(i: Int, j: Int) {
    val t = xs(i)
    xs(i) = xs(j)
    xs(j) = t
  }

  def sort(left: Int, right: Int) {
    val pivot = xs((left + right) / 2)
    var start = left
    var end = right
 
    while (start <= end) {
      while (xs(start) < pivot) start += 1
      while (xs(end) > pivot) end -= 1
      if (start <= end) {
        swap(start, end)
        start += 1
        end -= 1
      }
    }
    if (left < end) sort(left, end)
    if (end < right) sort(start, right)
  }

  sort(0, xs.length - 1)

}
