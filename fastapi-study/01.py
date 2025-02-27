from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.post('/123',
          tags=['这是123'],
          summary='这是总结，summary',
          description='这是详情，description',
          response_description='这是响应详情，responses',
          deprecated=True)
async def home():  # 路径操作函数
    return {'user_id': 1002}


if __name__ == '__main__':
    uvicorn.run('01:app', host='127.0.0.1', port=8080)
