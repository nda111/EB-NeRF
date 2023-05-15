import socket

CMD_ACQUIRE = 'acq'
CMD_QUIT = 'quit'

def send(sckt: socket.socket, data: str):
    return sckt.send(f'{data}\0'.encode())

def recv(sckt: socket.socket, buffer_size: int):
    return sckt.recv(buffer_size).decode().split('\0')[0]
