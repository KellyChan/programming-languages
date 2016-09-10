import ChecksumAccumulator.calculate

object Summer {
    def main(args: Array[String]){
        for (arg <- args)
            println(arg + ": " + calculate(arg))
    }
}

object FallWinterSpringSummer extends Application {
    for (season <- List("fall", "winter", "spring"))
        println(season + ": " + calculate(season))
}
