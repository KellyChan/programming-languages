val numbers1 = List(1,2)
val numbers2 = List(3,4)

val numbers = numbers1 ::: numbers2
println(numbers1 + " and " + numbers2 + " were not mutated.")
println("Thus, " + numbers + " is a new list.")

val numbers_append1 = 0 :: numbers2
val numbers_append2 = 5 :: numbers2
val numbers_appends = 0 :: numbers1 ::: numbers2 :: Nil
println(numbers_append1)
println(numbers_append2)
println(numbers_appends)
