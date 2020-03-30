
import subprocess
import time
import numpy as np
import socket
import threading


HOST = ''
PORT = 65432
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(0)


class env():
    filestate = "../state.txt"

    def __init__(self):
        self.state = []  # Include position of agent and position of enemy
        self.reward = 0
        self.dead = 0  # 1: is dead
        self.win = 0

    def step(self, action):
        # Send an action to client and get the information from java client
        conn, addr = s.accept()
        sess = Session(conn, action)
        # Handle the session in a separate thread.
        sess.start()
        # print(sess.get_state())
        return self.reward, self.state, self.dead

    def reset(self):
        _, state, _ = self.step(4)  # stay
        return(state)


class Session(threading.Thread):
    def __init__(self, conn, action):
        threading.Thread.__init__(self)
        self.conn = conn
        self.action = action

    def run(self):
        # Read a string from the client.

        # self.conn.sendall(str(self.action).encode())
        # agent = self.conn.recv(256)
        # enemy = self.conn.recv(256)
        # done = self.conn.recv(256)
        # win = self.conn.recv(256)
        # self.state = [agent, enemy, done, win]
        # print(agent)
        # We're done with this connection, so close it.
        # self.conn.sendall(agent)
        # self.conn.close()
        while True:
            # Read a string from the client.
            line = self.conn.recv(256)
            print(line)
            if line == '':
                # No more data from the client.  We're done.
                break

            # Convert the line to all caps and send it back to the client.
            self.conn.sendall(line.upper())

            # We're done with this connection, so close it.
        self.conn.close()

    def get_state(self):
        return self.state


if __name__ == "__main__":
    e = env()
    e.reset()
    while(True):
        e.step(1)
        time.sleep(0)
