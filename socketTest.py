from socket import*
import time

def step():
	print("step")
	return "ratios"

if __name__ == "__main__":
    #time.sleep(5) 
    print("python ----------------------")
    
    serverIP = '127.0.0.1'    # The remote host  
    serverPort = 54321           # The same port as used by the server  
    BUFSIZE = 1025
    ns3Server = (serverIP, serverPort)
    tcpSocket = socket(AF_INET, SOCK_STREAM)
    tcpSocket.connect(ns3Server)
    
    
    data = ""
    msgTotalLen = 0
    msgRecvLen = 0
    msg = ""
    count = 0
    blockSize = 1024;
    while True:
        datarecv = tcpSocket.recv(BUFSIZE).decode()
        if len(datarecv) > 0:
            #print("datarecv %s" % datarecv)
            if msgTotalLen == 0:
                totalLenStr = (datarecv.split(';'))[0]
                msgTotalLen = int(totalLenStr) + len(totalLenStr) + 1#1 is the length of ';'
            msgRecvLen += len(datarecv)
            msg += datarecv
            if msgRecvLen < msgTotalLen: 
                continue
            print(msg)
            print(len(msg))
            time.sleep(5) 
            #msg = "1234567890asgfasgagagdagahafhfdahsdddddaaa"
            #msg = str(len(msg)) + ';' + msg;
            #msgTotalLen = len(msg)
            #print(msgTotalLen)
            blockNum = int((msgTotalLen+blockSize-1)/blockSize);
            for i in range(blockNum):
                data = msg[i*blockSize:i*blockSize+blockSize]
                tcpSocket.send(data.encode())
                #print(data)
            data = ""
            msgTotalLen = 0
            msgRecvLen = 0
            msg = ""
            count += 1
            if count >= 2:
                break
    tcpSocket.close()
        
        
