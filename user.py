import socket


s = socket.socket()
port = 8888

count = 0

while (count!=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', port))
    print (s.recv(1024))
    a=input("Enter kro kuch")
    s.send(str(a))
    s.close()
