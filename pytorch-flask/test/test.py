import requests 

resp = requests.post('https://pytorch-flask-tut.herokuapp.com/predict', files={'file': open('three.png', 'rb')})
print(resp.text)