import subprocess

#Create class
subprocess.check_call(['javac','src/net/thedanpage/worldshardestgame/Game.java'])

#Copy class to file net
subprocess.call('mv src/net/thedanpage/worldshardestgame/*.class net/thedanpage/worldshardestgame',shell = True)

#Run game in file net
subprocess.call('java net.thedanpage.worldshardestgame.Game',shell=True)


