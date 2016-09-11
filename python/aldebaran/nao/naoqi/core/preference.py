import sys
import time
from naoqi *

NAO_IP = "nao.local"

Dummy = None
memory = None

# Create a dummy module, just to subscribe to events emitted by PreferenceManager
class DummyModule(ALModule):
    """ Dummy module """
    def __init__(self, name):
        ALModule.__init__(self, name)
        global memory
        memory = ALProxy("ALMemory")
        memory.subscribeToMicroEvent("preferenceAdded", "Dummy", "", "onPreferenceAdded")
        memory.subscribeToMicroEvent("preferenceChanged", "Dummy", "", "onPreferenceChanged")
        memory.subscribeToMicroEvent("preferenceRemoved", "Dummy", "", "onPreferenceRemoved")
        memory.subscribeToMicroEvent("preferenceDomainRemoved", "Dummy", "", "onPreferenceDomainRemoved")
    def onPreferenceAdded(self, event, value, message):
        print "Preference added: " + str(value)
    def onPreferenceChanged(self, event, value, message):
        print "Preference changed: " + str(value)
    def onPreferenceRemoved(self, event, value, message):
        print "Preference removed: " + str(value)
    def onPreferenceDomainRemoved(self, event, value, message):
        print "Preference domain removed: " + str(value)

def main():
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, NAO_IP, 9559)

    global Dummy
    Dummy = DummyModule("Dummy")

    # Play with preference manager
    pm = ALProxy("ALPreferenceManager")
    pm.setValue("com.apps.chess", "level", "hard")
    pm.setValue("com.apps.chess", "treedepth", "10")
    pm.setValue("foo.bar", "foo", "bar")
    pm.setValue("com.apps.chess", "level", "easy")
    print "com.apps.chess - level: " + pm.getValue("com.apps.chess", "level")
    pm.removeValue("foo.bar", "foo")
    pm.removeDomainValues("com.apps.chess")
    print "com.apps.chess - level: " + str(pm.getValue("com.apps.chess", "level"))

    time.sleep(1)
    myBroker.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()
