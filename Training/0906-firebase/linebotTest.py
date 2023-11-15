from flask import Flask
app = Flask(__name__)

from flask import request, abort
from linebot import  LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from firebase import firebase
import ast

line_bot_api = LineBotApi('544wxSt4FvUzrdDahdDpWLmTV3RIJtnXEKABPhY4YHpXeguLe7B0yEd+ct7QJ8oeUnrILwDwYvrHVKkhxcDVt3yl63rgiyEgYvSId/8abuVqnsXeiuEEe16qwXCEg1Rykn7sM2kRpbGKUTdprh/QpgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('a7755698ee49077692b8ec25e7cc87e0')

url = 'https://cgu-db-547f1-default-rtdb.firebaseio.com/'
fb = firebase.FirebaseApplication(url,None)

userlist= [{"name":"Bert"},{"name":"Lily"}]
for user in userlist:
    fb.post('userlist',user)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    mtext = event.message.text
    if mtext == '@help':
        try:
            message = TextSendMessage(text = "請問有什麼可以幫助您")
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '@report':
        getlist = fb.get("/userlist/",None)
        name_str = ""
        for key,value in getlist.items():
            name_str = name_str + value["name"] + "\n"
        try:
            message = TextSendMessage(text = name_str.rstrip())
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
        
    elif mtext[:5] == '@add@':
        addlist = [{"name":mtext[5:]}]
        for newname in addlist:
            fb.post('userlist', newname)
        try:
            message = TextSendMessage(text = "新增完成！")
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text="發生錯誤！"))

    elif mtext[:8] == '@remove@':
        getlist = fb.get("/userlist/",None)
        for key,value in getlist.items():
            if value["name"] == mtext[8:]:
                fb.delete('/userlist/',key)
        try:
            message = TextSendMessage(text = "刪除完成！")
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text="發生錯誤！"))


if __name__ == '__main__':
    app.run()
