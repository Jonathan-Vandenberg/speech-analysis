from fastapi.testclient import TestClient
from pathlib import Path
from main.app import app

client = TestClient(app)
path = Path('tests/audio/regression/basic_match.wav')
files = {'file': (path.name, path.read_bytes(), 'audio/wav')}
data = {'expected_text': 'Hello world.'}
res = client.post('/analyze/pronunciation', data=data, files=files, headers={'Authorization':'Bearer dummy'})
print(res.status_code)
print(res.json())
