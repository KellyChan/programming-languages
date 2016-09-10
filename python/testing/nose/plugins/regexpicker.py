import sys
err = sys.stderr

import re

import nose
from nose.plugins import Plugin

class RegexPicker(Plugin):

    name = "regexpicker"

    def __init__(self):
        Plugin.__init__(self)
        self.verbose = False

    def options(self, parser, env):
        Plugin.options(self, parser, env)
        parser.add_option("--re-pattern", \
                          dest="pattern", \
                          action="store", \
                          default=env.get("NOSE_REGEX_PATTERN", "test.*"), \
                          help=("Run test methods that have a method name \
                                 matching this regular expression"))

    def configure(self, options, conf):
        Plugin.configure(self, options, conf)
        self.pattern = options.pattern
        if options.verbosity >= 2:
            self.verbose = True
            if self.enabled:
                err.write("Pattern for matching test methods is %s\n" % self.pattern)

    def wantMethod(self, method):
        wanted = re.match(self.pattern, method.func_name) is not None
        if self.verbose and wanted:
            err.write("nose will run %s\n" % method.func_name)
        return wanted

