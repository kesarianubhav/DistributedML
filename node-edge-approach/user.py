import socket

def User(ip='127.0.0.1',port=8887):

    s = socket.socket()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', port))
    s.send(b"Mujhe Kaam De do !!")
    while(1):
        print (s.recv(1024).decode("ASCII"))
        a=input("Abe Enter kr naa kuch >>").encode('ASCII')
        s.send(a)
    s.close()


if __name__=="__main__":
    User()
