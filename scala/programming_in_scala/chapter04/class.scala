class ChecksumAccumulator {

    private var sum = 0

    def add(b: Byte): Unit = {
        sum += b
    }

    def checksum(): Int = {
        return ~(sum & 0xFF) + 1
    }
}


//----------------------------------------------------

class ChecksumAccumulator {
    private var sum = 0
    def add(b: Byte): Unit = sum += b
    def checksum(): Int = ~(sum & 0xFF) + 1
}

//----------------------------------------------------


class ChecksumAccumulator {
    private var sum = 0
    def add(b: Byte) { sum += b }
    def checksum(): Int = ~(sum & 0xFF) + 1
}
