from requests import session
from transformers import AutoModelForCausalLM, AutoTokenizer
import eventlet
import socketio
import torch
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(dev)
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

# Let's chat for 5 lines
class Leo:
    def __init__(self, sid):
        self.user = sid
        self.history = []
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")#.to(dev)
        self.step = 0
        self.chat_history_ids = 0

    def __del__(self):
        print("Good bye my friend")

    def message(self,msg):

    # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(msg+self.tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.step > 0 else new_user_input_ids
    # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = self.model.generate(bot_input_ids,
                                      max_length=1000,
                                      do_sample=True,
                                      top_p=0.95,
                                      top_k=0,
                                      temperature=0.75,
                                      pad_token_id=self.tokenizer.eos_token_id
                                      )

    # pretty print last ouput tokens from bot
        self.step+=1
        return self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


sessions      = []

@sio.event
def connect(sid, environ):
    print('connect', sid)
    sessions.append(Leo(sid))
    # sio.emit('message', { 'user' : "Leo" , 'msg': 'hello' }, room=sid)

@sio.event
def message(sid, data):
    print(f'($): {data}')
    for session in sessions:
        if session.user == sid:
            response = session.message(data['msg'])
        else: 
            pass
    print(response)
    sio.emit('message', { 'user' : "Leo" , 'msg': response }, room=sid)


@sio.event
def disconnect(sid):
    for session in sessions:
        if session.user == sid:
            sessions.remove(session)
    print(sessions)
    print(f'bye user {sid}')
    

eventlet.wsgi.server(eventlet.listen(('',5000)), app)