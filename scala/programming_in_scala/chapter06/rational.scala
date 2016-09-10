class Rational(n: Int, d: Int) {

    require(d != 0)
    
    private val g = gcd(n.abs, d.abs)
    val numer = n / g
    val denom = d / g
    
    def this(n: Int) = this(n, 1)
    def add(that: Rational): Rational = 
        new Rational(
            numer * that.denom + that.numer * denom,
            denom * that.denom
        )
       
        override def toString = numer + "/" + denom

        private def gcd(a: Int, b: Int): Int = 
            if (b==0) a else gcd(b, a % b)
}
