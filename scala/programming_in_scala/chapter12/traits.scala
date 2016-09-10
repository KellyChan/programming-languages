trait Philosophical {
  def philosophize() {
    println("I consume memory, therefore I am!")
  }
}


class Animal
trait HasLegs

class Frog extends Animal with Philosophical with HasLegs {
  override def toString = "green"

  override def philosophize() {
    println("It ain't easy being " + toString + "!")
  }
}
