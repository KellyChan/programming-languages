object Predef {
  type Map[A, +B] = collection.immutable.Map[A, B]
  type Set[A] = collection.immutable.Set[A]
  
  val Map = collection.immutable.Map
  val Set = collection.immutable.Set
}
