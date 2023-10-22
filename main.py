import uvicorn
import os
#from run_api import load_setting

# 读取运行环境配置
#load_setting()

if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        port=8080,
        host='0.0.0.0')