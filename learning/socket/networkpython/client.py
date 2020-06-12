#!/usr/bin/python						

import socket								# Import socket module
import time

s = socket.socket()					# Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345								# Reserve a port for your service.

s.connect((host, port))
for i in range(10):
	print s.recv(2)
	s.send('abtf')
s.close()				
