import socket
import time


HEADERSIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 7))
s.listen(1)

try:
	while True:
			# now our endpoint knows about the OTHER endpoint.
			clientsocket, address = s.accept()
			print(f"Connection from {address} has been established.")

			msg = "Welcome to the server!"
			msg = f"{len(msg):<{HEADERSIZE}}"+msg
			while True:
				line = clientsocket.recv(16)
				print(line)
				if line == "":
					break
				clientsocket.send(msg.encode())
finally:
	client.socket.close()
