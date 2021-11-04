# IERG3310 Project
# first version written by FENG, Shen Cody
# second version modified by YOU, Lizhao
# Third version modified by Jonathan Liang @ 2016.10.25

import socket
import random
import numpy as np
import time 

def generate_random_str(randomlength):
    random_str = ''
    base_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwnxyz1234567890'
    length = len(base_str) - 1
    for i in range(int(randomlength)):
        random_str += base_str[random.randint(0,length)]
    return random_str

robotVersion = "3.0"
listenPort = 3310
socket.setdefaulttimeout(120)
localhost = ''

print ("Robot version " + robotVersion + " started")
print ("You are reminded to check for the latest available version")

print ("")

# Create a TCP socket to listen connection
print ("Creating TCP socket...")
listenSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listenSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listenSocket.bind((localhost, listenPort))
listenSocket.listen(5)
print ("Done")

print ("\nTCP socket created, ready for listening and accepting connection...")
#print "Waiting for connection on port %(listenPort)s" % locals()
print ("Waiting for connection on port", listenPort)

# accept connections from outside, a new socket is constructed
s1, address = listenSocket.accept()
studentIP = address[0]
print ("\nClient from %s at port %d connected" %(studentIP,address[1]))
# Close the listen socket
# Usually you can use a loop to accept new connections
listenSocket.close()

iTCPPort2Connect = random.randint(0,9999) + 20000
print ("Requesting STUDENT to accept TCP <%d>..." %iTCPPort2Connect)

s1.send(str(iTCPPort2Connect).encode())
print ("Done")

time.sleep(1)
print ("\nConnecting to the STUDENT s1 <%d>..." %iTCPPort2Connect)
############################################################################# phase 0
# Connect to the server (student s2)

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.connect((studentIP,iTCPPort2Connect))
bs = s2.recv(10)
bs = bs.decode()
bs = bs[2:]
print("The receiver's buffer size is :", bs)


message = generate_random_str(40000)
print(len(message))

# Receive the buffer size

s2.send(message.encode())


print("Done")


s1.close()
s2.close()
exit()
