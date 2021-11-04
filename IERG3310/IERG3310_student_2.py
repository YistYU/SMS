# IERG3310 Project
# first version written by FENG, Shen Cody
# second version modified by YOU, Lizhao
# Third version modified by Jonathan Liang @ 2016.10.25

import socket
import random
import time
import struct
import datetime


HOST = '192.168.50.146'  # The server's hostname or IP address
PORT = 3310      # The port used by the server

def recvall(sock):
    BUFF_SIZE = 1000 # 4 KiB
    count = 0
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        #print("Number of received messages:", len(part), "total received bytes, ",len(data))
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data


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
bufsize = s_2.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)

print( "Buffer size is: ", bufsize)

bs_data = 'bs' + str(bufsize)
s_2.send(bs_data.encode())

# Finish to send the buffer size
a = time.time()
data = recvall(s_2)
b = time.time()
gap = (int(round((b-a)*1000)))
print("Time gap is ",gap)
print(len(data.decode()))

s_1.close()
s_2.close()

print("Finished!")

exit()
