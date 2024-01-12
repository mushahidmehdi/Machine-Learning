
import requests

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

resp = requests.get(url)

with open('file.txt', 'wb') as f:
    f.write(resp.content)
