import socket
import threading
 
# Listen for TCP connections on port 65432 on any interface.
HOST = ''
PORT = 65432
 
def main():
	# Create a TCP server socket.
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((HOST, PORT))
	s.listen(1)
 
	while True:
		# Wait for a connection from a client.
		conn, addr = s.accept()
 
		# Handle the session in a separate thread.
		Session(conn).start();
 
# Session is a thread that handles a client connection.
class Session(threading.Thread):
	def __init__(self, conn):
		threading.Thread.__init__(self)
		self.conn = conn
 
	def run(self):
		while True:
			# Read a string from the client.
			line = self.conn.recv(256)
			if line == '':
				# No more data from the client.  We're done.
				break
 
			# Convert the line to all caps and send it back to the client.
			self.conn.sendall(line.upper())
 
		# We're done with this connection, so close it.
		self.conn.close()
 
if __name__ == '__main__':
	main()
