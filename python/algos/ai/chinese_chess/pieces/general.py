"""
Chinese Chess
=====================================================================

Pieces: General/Marshal

Rules:
  - General: {d8, d9, d10, e8, e9, e10, f8, f9, f10}
  - Marshal: {d1, d2, d3, e1, e2, e3, f1, f2, f3}
  - General & Marshal: {{d,e}, {d,f}, {e,f}}

Board:

    10 +-+-+-*-*-*-+-+-+
    09 +-+-+-*-*-*-+-+-+
    08 +-+-+-*-*-*-+-+-+
    07 +-+-+-+-+-+-+-+-+
    06 +-+-+-+-+-+-+-+-+
    05 +-+-+-+-+-+-+-+-+
    04 +-+-+-+-+-+-+-+-+
    03 +-+-+-*-*-*-+-+-+
    02 +-+-+-*-*-*-+-+-+
    01 +-+-+-*-*-*-+-+-+
       a b c d e f g h i


"""

MARSHAL_LOCATIONS = ['d1', 'd2', 'd3',
                     'e1', 'e2', 'e3',
                     'f1', 'f2', 'f3']
GENERAL_LOCATIONS = ['d8', 'd9', 'd10',
                     'e8', 'e9', 'e10',
                     'f8', 'f9', 'f10']

class General(object):

    def __init__(self):
        #self.marshal_location = marshal_location
        pass

    def possible_moves(self):
        """
        input: (Marshal) d1
        outputs: (General) {e*,f*} 

            {
                'd*': {e*, f*},
                'e*': {d*, f*},
                'f*': {d*, e*}
            }

        outputs: [(d1, e10), (d1, e9), ...]
        """

        moves = []

        for mloc in MARSHAL_LOCATIONS:
            if mloc.startswith('d'):
                glocs = [(mloc, gloc) for gloc in GENERAL_LOCATIONS if gloc.startswith('e') or gloc.startswith('f')]
                moves.extend(glocs)
            elif mloc.startswith('e'):
                glocs = [(mloc, gloc) for gloc in GENERAL_LOCATIONS if gloc.startswith('d') or gloc.startswith('f')]
                moves.extend(glocs)
            elif mloc.startswith('f'):
                glocs = [(mloc, gloc) for gloc in GENERAL_LOCATIONS if gloc.startswith('d') or gloc.startswith('e')]
                moves.extend(glocs)
                
        return moves

    def move(self):
        pass
