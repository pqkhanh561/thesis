import socket	
import time	
						# Import socket module

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)					# Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345	
print host							# Reserve a port for your service.
s.bind((host, port))				# Bind to the port

s.listen(5)									# Now wait for client connection.
c, addr = s.accept()			# Establish connection with client.
print 'Got connection from', addr
for i in range(50):
		c.send('up   ')
		tmp =''
		while True:
			t = c.recv(16)
			print("t")
			tmp = tmp + t
			if tmp.count(',')>=3 and tmp[-1]!=',':
				break
		


while True:
		c.send('up   ')
		tmp =''
		while True:
			t = c.recv(16)
			tmp =tmp + t
			if tmp.count(',')>=11 and tmp[-1]!=',':
				break
		if not len(t):
			break
		#print tmp
c.close()								# Close the connection
