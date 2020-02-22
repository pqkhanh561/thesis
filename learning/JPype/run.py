# Import module
import jpype

# Enable Java imports
import jpype.imports
from jpype import *
# Pull in types

# Launch the JVM
jpype.addClassPath("/net/thedanpage/worldshardestgame/getProperty/*")
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")

from java.lang import System
print(System.getProperty("net.thedanpage.worldshardestgame"))
