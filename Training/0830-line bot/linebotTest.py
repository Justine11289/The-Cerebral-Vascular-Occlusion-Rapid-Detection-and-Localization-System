from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from flask import request, abort
from flask import Flask
app = Flask(__name__)

# Channel access token
line_bot_api = LineBotApi(
    '544wxSt4FvUzrdDahdDpWLmTV3RIJtnXEKABPhY4YHpXeguLe7B0yEd+ct7QJ8oeUnrILwDwYvrHVKkhxcDVt3yl63rgiyEgYvSId/8abuVqnsXeiuEEe16qwXCEg1Rykn7sM2kRpbGKUTdprh/QpgdB04t89/1O/w1cDnyilFU=')
# Channel secret
handler = WebhookHandler('a7755698ee49077692b8ec25e7cc87e0')


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
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text))


if __name__ == '__main__':
    app.run()
