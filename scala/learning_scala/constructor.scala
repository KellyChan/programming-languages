// immutable

//class Greeter(message: String) {
//  println("A greeter is being instantiated")

//  def SayHi() = println(message)
//}

//val greeter = new Greeter("Hello world!")
//greeter.SayHi()

//---------------------------------------------------------------------
// mutable

//class Greeter(var message: String) {
//  println("A greeter is being instantiated")

//  message = "I was asked to say " + message
//  def SayHi() = println(message)
//}

//val greeter = new Greeter("Hello world!")
//greeter.SayHi()

//---------------------------------------------------------------------

//class Greeter(message: String, secondaryMessage: String) {
//  def this(message: String) = this(message, "")
//  def SayHi() = println(message + secondaryMessage)
//}

//val greeter = new Greeter(
//    "Hello world!", 
//    " I'm a bit more chatty than my predecessors.")
//greeter.SayHi()

//---------------------------------------------------------------------
// abstract class

abstract class Greeter {
    val message: String
    def SayHi() = println(message)
}

class SwedishGreeter extends Greeter{
    message = "Hej världen!"
}

val greeter = new SwedishGreeter()
greeter.SayHi()
