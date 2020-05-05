import socket

host = ''        # Symbolic name meaning all available interfaces
port = 65432# Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

print(host , port)
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
while True:

    try:
        data = conn.recv(256)

        if not data: break

        print("Client Says: "+data.decode())
        conn.sendall("Server Says:hi".encode())

    except socket.error:
        print("Error Occured.")
        break

conn.close()
