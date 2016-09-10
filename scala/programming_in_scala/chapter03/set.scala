var jetSet = Set("Boeing", "Airbus")

jetSet += "Lear"
println(jetSet.contains("Cessna"))
println(jetSet.contains("Lear"))

//---------------------------------------------------

import scala.collection.mutable.Set

val movieSet = Set("Hitch", "Poltergeist")

movieSet += "Shrek"
println(movieSet)

//---------------------------------------------------

import scala.collection.immutable.HashSet

val hashSet = HashSet("Tomatoes", "Chilies")

println(hashSet + "Coriander")
