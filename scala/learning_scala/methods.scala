class TextHelper {
  val suffix = "..."

  def ellipse(original: String, maxLength: Int): String = {
    if (original.length <= maxLength)
      return original;
    original.substring(0, maxLength - suffix.length) + suffix;
  }
}

val helper = new TextHelper()
println(helper.ellipse("Hello world!", 10))
