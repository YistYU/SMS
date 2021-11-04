# IERG3310 Project
# first version written by FENG, Shen Cody
# second version modified by YOU, Lizhao
# Third version modified by Jonathan Liang @ 2016.10.25

import socket
import random
import time
import struct


HOST = '192.168.50.146'  # The server's hostname or IP address
PORT = 3310      # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_1:
    s_1.connect((HOST, PORT))
    s_1.sendall(b'1155141476')
    data = s_1.recv(5)

print('Received', repr(data))

PORT = int(data)

print ("Creating TCP socket...")
listenSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listenSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listenSocket.bind((HOST, PORT))
listenSocket.listen(5)
print ("Done")

print ("\nTCP socket created, ready for listening and accepting connection...")
#print "Waiting for connection on port %(listenPort)s" % locals()
print ("Waiting for connection on port", PORT)

s_2, address = listenSocket.accept()
data = s_2.recv(12)
string = data.decode().split(",")
#print(string)
port1 = string[0]
port2 = string[1].split(".")[0]
print(port1, port2)

num =  str(random.randint(5, 10))
PORT = int(port1)

ADDR = ('', PORT)
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_3:
    s_3.sendto(num.encode(), ADDR)
PORT = int(port2)

listenSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listenSocket.bind(('', PORT))

print ("\nUDP socket created, ready for listening and accepting connection...")
#print "Waiting for connection on port %(listenPort)s" % locals()
print ("Waiting for connection on port", PORT)

data, address = listenSocket.recvfrom(1024)
PORT = int(port1)

if data:
    for i in range(0,5):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_4:
            s_4.sendto(data, ADDR)
        time.sleep(1)
        print ("UDP packet %d sent" %(i+1))
    
s_1.close()
s_2.close()
s_3.close()
s_4.close()

print("Finished!")

exit()
