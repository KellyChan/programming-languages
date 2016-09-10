// Returns a row as a sequence
def makeRowSeq(row: Int) = 
  for (col <- 1 to 10) yield {
    val prod = (row * col).toString
    val padding = " " * (4 - prod.length)
    padding + prod  
  }

// Returns a row as a string
def makeRow(row: Int) = makeRowSeq(row).mkString

// Returns table as a string with one row per line
def multiTable() = {
  val tableSeq = // a sequence of row strings
    for (row <- 1 to 10)
    yield makeRow(row)
  tableSeq.mkString("\n")
}

println(multiTable())
