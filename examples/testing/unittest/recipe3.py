
class RomanNumeralConverter(object):

    def __init__(self):
        self.digit_map = {
                          "M": 1000,
                          "D": 500,
                          "C": 100,
                          "L": 50,
                          "X": 10,
                          "V": 5,
                          "I": 1
                         }

    def convert_to_decimal(self, roman_numeral):
        val = 0
        for char in roman_numeral:
            val += self.digit_map[char]
            if val > 4000:
                raise Exception("We don't handle values over 4000")
            return val

    def convert_to_roman(self, decimal):
        if decimal > 4000:
            raise Exception("We don't handle values over 4000.")
        
        val = ""
        mappers = [
                   (1000, "M"),
                   (500, "D"),
                   (100, "C"),
                   (50, "L"),
                   (10, "X"),
                   (5, "V"),
                   (1, "I")
                  ]

        for (mapper_dec, mapper_rom) in mappers:
            while decimal >= mapper_dec:
                val += mapper_rom
                decimal -= mapper_dec
            return val

