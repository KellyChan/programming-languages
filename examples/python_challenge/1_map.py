from string import maketrans

alphabet = "abcdefghijklmnopqrstuvwxyz"
alphabet_decode = "cdefghijklmnopqrstuvwxyzab"
map_table = maketrans(alphabet, alphabet_decode)


if __name__ == '__main__':

    strings = """
    g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj.
    """

    url = "map"

    print strings.translate(map_table)
    print url.translate(map_table)
