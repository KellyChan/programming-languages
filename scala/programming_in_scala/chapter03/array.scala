val big = new java.math.BigInteger("12345")

// val greetStrings = new Array[String](3)
val greetStrings: Array[String] = new Array[String](3)
greetStrings(0) = "Hello "
greetStrings(1) = "world"
greetStrings(2) = "!"

greetStrings.update(1, "Alice")

for (i <- 0 to 2)
  print(greetStrings(i))

