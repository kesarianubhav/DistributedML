import requests

# api-endpoint
URL = 'http://localhost:9999'

# location given here
#location = "delhi technological university"

# defining a params dict for the parameters to be sent to the API
PARAMS = {
        'start':1, 'stop':5,'step':1
            }

# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS)

# extracting data in json format
data = r.json()

print(data)
