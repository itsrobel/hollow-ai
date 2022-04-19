import socketio

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')
    sio.emit("message", get_input())
    sio.wait()


@sio.event
def message(data):
    print('message received with ', data)
    sio.emit("message", get_input())
    # sio.emit('my response', {'response': 'my response'})


@sio.event
def disconnect():
    print('disconnected from server')


def get_input():
    return str(input("($): "))


sio.connect('http://localhost:5000')
# connected = True
# while connected:
#     new_msg = str(input("($): "))
#     if new_msg == "exit":
#         connected = False
#     sio.emit("my_message",new_msg )
#     sio.wait()
