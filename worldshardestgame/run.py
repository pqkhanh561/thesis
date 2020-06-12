#!/usr/bin/python3.7
import subprocess

#Create class
<<<<<<< HEAD

subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/GameLevel.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Game.java'])
=======
>>>>>>> 4dc3963957bdabb2f14aef9934dd7186837db832

subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Player.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Input.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/TextFileWriter.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Game.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/GameLevel.java'])
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Dot.java'])
#Copy class to file net
subprocess.call('mv src/net/thedanpage/worldshardestgame/*class net/thedanpage/worldshardestgame',shell = True)

#Run game in file net
subprocess.call('java net.thedanpage.worldshardestgame.Game',shell=True)


