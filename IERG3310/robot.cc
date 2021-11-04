/*
 * IERG3310 Project
 * first version written by FENG, Shen Cody
 * Modified by Jonathan Liang @ 2016.10.25
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define LISTEN_PORT 3310
#define ROBOT_VERSION "3.0"

int main(){

	char messageBuffer[1024];
	char messageBuffer2[1024];
	char* studentIP;
	int messageBufferSize = sizeof(messageBuffer);
	int ListenSocket, s1, s2, s3;

	printf("Robot version %s (Oct. 24, 2016) started \n", ROBOT_VERSION);
	printf("You are reminded to check for the latest available version\n\n");

	// --------------------------------------------------------------------- //
	//								Step 1/2								 //
	// --------------------------------------------------------------------- //
	// Create a SOCKET for listening for incoming connection requests.
	printf("Creating TCP socket for listening and accepting connection...");
	ListenSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (ListenSocket < 0) {
		printf(" %s %d \n",strerror(errno), errno);
		printf("Error at creating socket...\n");
		return 1;
	}

	// The sockaddr_in structure specifies the address family,
	// IP address, and port for the socket that is being bound.
	struct sockaddr_in service;
	service.sin_family = AF_INET;
	service.sin_addr.s_addr = inet_addr("0.0.0.0");
	service.sin_port = htons(LISTEN_PORT);

	if (bind( ListenSocket, (struct sockaddr *) &service, sizeof(service)) < 0) {
		printf(" %s %d \n",strerror(errno), errno);
		printf("bind() failed.\n");
		close(ListenSocket);
		return 1;
	}
	if (listen( ListenSocket, 1 ) < 0) {
		printf(" %s %d \n",strerror(errno), errno);
		printf("Error listening on socket.\n");
		close(ListenSocket);
		return 1;
	}
	printf("Done\n");

	printf("\nReady to accept connection on port %d\n", LISTEN_PORT);
	printf("Waiting for connection...\n");
	// When accept is called, the function is blocked and wait until a client is connected
	// A new socket is returned when accepting a connection

	struct sockaddr_in sin; 
	int size = sizeof(sin);
	memset(&sin, 0, size);

	s1 = accept( ListenSocket, (struct sockaddr*)&sin, (socklen_t *)&size );
	if (s1 < 0) {
		printf(" %s %d \n",strerror(errno), errno);
		printf("accept failed\n");
		close(ListenSocket);
		return 1;
	}
	//	The listen socket is no longer required
	//	Usually you can use a loop to accept new connections
	close(ListenSocket);

	printf("\nClient from %s at port %d connected\n", inet_ntoa(sin.sin_addr), ntohs(sin.sin_port));
	studentIP = inet_ntoa(sin.sin_addr);
	
	char studentID[11];
	memset(studentID, 0, 11);
	int totalByteReceived = 0;
	int byteToReceive = 10;
	int iResult=0;

	//	Receive the data from the socket s1
	//	TCP socket is a stream socket which does not preserve message boundary
	//	Calling recv to receive k bytes may receive less than k bytes
	//	A loop is required
	do {
		iResult = recv(s1, studentID+totalByteReceived, 11-totalByteReceived, 0);
		if ( iResult > 0 ){
			printf("Bytes received: %d\n", iResult);
			totalByteReceived += iResult;
			if (totalByteReceived >= byteToReceive)
				break;
		}
		else if ( iResult == 0 ){
			printf(" %s %d \n",strerror(errno), errno);
			printf("Connection closed\n");
			return 1;
		}
		else{
			printf(" %s %d \n",strerror(errno), errno);			
			printf("recv failed\n");
			return 1;
		}
	} while( iResult > 0 );
	printf("Stduent ID received: %s\n", studentID);
    
	// --------------------------------------------------------------------- //
	//								Step 3									 //
	// --------------------------------------------------------------------- //
	// Generate a random port number and ask STUDENT to listen
	srand((unsigned)time( NULL ));
	int iTCPPort2Connect = rand()%10000 + 20000;

	printf("Requesting STUDENT to accept TCP <%d>...", iTCPPort2Connect);
	// Send the port required to the STUDENT
	memset(messageBuffer, 0, messageBufferSize);
	sprintf(messageBuffer, "%d", iTCPPort2Connect);

	int byte_left = (int)strlen(messageBuffer);
	// Similar to recv, a loop is required
	while (byte_left > 0){
		iResult = send( s1, messageBuffer-byte_left+(int)strlen(messageBuffer), byte_left, 0 );

		if (iResult < 0) {
			printf("send() failed with error\n");
			return 1;
		}

		byte_left -= iResult;
	}
	printf("Done\n"); 

	// --------------------------------------------------------------------- //
	//								Step 4									 //
	// --------------------------------------------------------------------- //
	sleep(1);
	printf("\nConnecting to the STUDENT s2 <%d> ...", iTCPPort2Connect);
	struct sockaddr_in clientService; 
	clientService.sin_family = AF_INET;
	clientService.sin_addr.s_addr = inet_addr(studentIP);
	clientService.sin_port = htons(iTCPPort2Connect);

	// Connect to server.
	if(	(s2 = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		printf("Error at creating socket s2 ... \n");
		return 1;
	}
	if ( connect( s2, (struct sockaddr *) &clientService, sizeof(clientService) ) < 0) {
		printf( "Error: connection failed.\n");
		printf(" %s %d \n",strerror(errno), errno);
		close(s2);
		return 1;
	}
	printf("Done\n");

	// Send the ports required to the STUDENT
	int iUDPPortRobot = rand()%10000 + 20000;
	int iUDPPortStudent = rand()%10000 + 20000;
	printf("Sending the UDP information: to ROBOT: <%d>, to STUDENT: <%d>...", iUDPPortRobot, iUDPPortStudent);
	memset(messageBuffer, 0, messageBufferSize);
	sprintf(messageBuffer, "%d,%d.", iUDPPortRobot, iUDPPortStudent);

	byte_left = (int)strlen(messageBuffer);
	while (byte_left > 0){
		iResult = send( s2, messageBuffer-byte_left+(int)strlen(messageBuffer), byte_left, 0 );

		if (iResult < 0) {
			printf("send() failed with error\n");
			return 1;
		}

		byte_left -= iResult;
	}
	printf("Done\n");

	// Create a UDP socket to send the data
	printf("\nPreparing socket s3 <%d>...", iUDPPortStudent);
	s3 = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s3 < 0) {
		printf("Error at socket()\n");
		return 1;
	}
	struct sockaddr_in udpAddr;
	udpAddr.sin_family = AF_INET;
	udpAddr.sin_port = htons(iUDPPortRobot);
	udpAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	bind(s3, (struct sockaddr *) &udpAddr, sizeof(udpAddr));
	
	//	Prepare the receiver address for the UDP packet
	struct sockaddr_in studentAddr;
	studentAddr.sin_family = AF_INET;
	studentAddr.sin_port = htons(iUDPPortStudent);
	studentAddr.sin_addr.s_addr = inet_addr(studentIP);

	printf("Done\n");

	// --------------------------------------------------------------------- //
	//								Step 5									 //
	// --------------------------------------------------------------------- //
	// Calling recv to receive a UDP packet without considering the sender
	// Calling recvfrom allows programmer to get more information about the packet received 
	// receive the value x and send x*10 bytes to student 
	memset(messageBuffer2, 0, messageBufferSize);
	recv(s3, messageBuffer2, messageBufferSize, 0);
	/* convert char to int for finding out x; x must be > 5 and < 10 */
	int x = 0; 
	for (int i = 0; messageBuffer2[i] != '\0' && i < 2; i++) 
		x = x * 10 + messageBuffer2[i] - '0'; /* messageBuffer[2] is a character */

	if (x <= 5) { 
		printf("at least 60 bytes are required...\n");
		close(s3);
		return 1; 
	}

	printf("Sending UDP packets: %d bytes...\n", x * 10);

	//	Random string to send
	memset(messageBuffer, 0, messageBufferSize);
	for (int i = 0; i < 2*x; i++)
		snprintf(messageBuffer+i*5, messageBufferSize-i*5, "%05d", rand()%10000);
	printf("Message to transmit: %s\n", messageBuffer);

	for (int i=0; i<5; i++){
		int t = sendto(s3, messageBuffer, strlen(messageBuffer), 0, (struct sockaddr*) &studentAddr, sizeof(studentAddr));
		sleep(1);
		printf("UDP packet %d sent\n", i+1);
	}

	printf("\nReceiving UDP packet:\n");

	memset(messageBuffer2, 0, messageBufferSize);
	recv(s3, messageBuffer2, messageBufferSize, 0);
	printf("Received: %s\n", messageBuffer2);

	if (strcmp(messageBuffer, messageBuffer2) == 0)
		printf("\nThe two strings are the same.\n");
	else
		printf("\nThe two strings are not the same.\n");

	close(s1);
	close(s2);
	close(s3);
		
	return 0;
}
