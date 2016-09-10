import nose

from plugins.regexpicker import RegexPicker
from plugins.csvreport import CsvReport

if __name__ == "__main__":

    nose.run(argv=["", "test_shoppingcart", "--verbosity=2"])

    args = ["", "test_shoppingcart", "--with-regexpicker", \
                                     "--re-pattern=test.*|length", \
                                     "--verbosity=2"]
    print "With verbosity..."
    print "==========================="
    nose.run(argv=args, plugins=[RegexPicker()])
    print "Without verbosity..."
    print "==========================="
    args = args[:-1]
    nose.run(argv=args, plugins=[RegexPicker()])

    
    args = ["", "test_shoppingcart", "--with-csv-report", "--csv-file=test_shoppingcart.csv"]
    nose.run(argv=args, plugins=[CsvReport()])


