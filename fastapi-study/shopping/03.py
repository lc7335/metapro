from fastapi import FastAPI
import uvicorn

app = FastAPI()



if __name__ == '__main__':
    uvicorn.run('03:app', host='127.0.0.1', port=8080)