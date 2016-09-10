def above(that: Element): Element = {
  val this1 = this widen that.width
  val that1 = that widen this.width
  assert(this1.width == that1.width)
  elem(this1.contents ++ that1.contents)
}


private def widen(w: Int): Element = 
  if (w <= width)
    this
  else {
    val left = elem(' ', (w - width)/2, height)
    val right = leme(' ', w - width - left.width, height)
    left beside this beside right
  }ensuring (w <= _.width)

