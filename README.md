# FastAPI Code AI Complete

使用 `FastAPI` 搭配預訓練 `CodeGeeX2` 模型，通過 AI 智能生成代碼

> 目前測試起來，使用英文自然語言來描述代碼表現比較良好，用中文會給你寫一些比較奇怪甚至不相關的功能出來。
> 總的來說，**功能還是很強大**，目前正在整合 `CodeGeeX2-6B` + `ChatGLM3-6B` 量化模型，使兩個偉大的作品很好的被玩轉！

建置 **Docker** 鏡像來打包整個 **API** 服務成一個程序 *(包含模型文件大小 > `40G` )*

---

## 🎥 效果演示

- `AI` 依指定需求生成代碼 🔗 http://127.0.0.1:7860/ai/copilot/coding

  ![ai-coding](https://github.com/AllennLiu/fastapi-code-aicomplete/assets/27174570/2978ffa4-e08b-41d7-882e-f83c7011453e)

---

## 🌐 環境準備

- 本項目已經將已配置好的環境鏡像，推在 `Dockerhub` 上了，**大小 `27G`** *(內含已載入的量化 `CodeGeeX2-6B` + `ChatGLM3-6B` 模型)*
- 鏡像連結：[seven6306/pretrained-model:ai-fastapi-copilot](https://hub.docker.com/repository/docker/seven6306/pretrained-model/tags)
- 如果您要手動 Build **Docker** 鏡像 `ai-fastapi-copilot` *(使用 Docker 來避免環境依賴等問題)*

  ```bash
  docker build --no-cache -t seven6306/pretrained-model:ai-fastapi-copilot . \
    --build-arg "HTTP_PROXY=http://admin:ZD7EdEpF9qCYpDpu@10.99.104.250:8081/" \
    --build-arg "HTTPS_PROXY=http://admin:ZD7EdEpF9qCYpDpu@10.99.104.250:8081/" \
    --build-arg "NO_PROXY=localhost,127.0.0.1,.example.com"
  # 掛代理 --build-arg 是我內部環境用的，可以刪除
  ```

---

## 🚀 快速開始

- 啟動 **API** 服務，這裡啟用了 **GPU Driver** 請確保安裝 Docker 的系統環境已安裝 `cuda-toolkit`，如果用 CPU 請刪除 `--gpus all` 與 `-e "LOAD_MODEL_DEVICE=gpu"` 兩個參數。

  ```bash
  docker run -tid -p 7861:22 -p 7860:7860 \
    --gpus all \
    -e "LOAD_MODEL_DEVICE=gpu" \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /root/.ssh:/root/.ssh:ro \
    --ulimit nofile=65535 --privileged=true --restart=always \
    --name copilot seven6306/pretrained-model:ai-fastapi-copilot
  ```

- 生成第一個 `Python` 腳本 *(沒有 `jq` 工具的話，需自行將 response 文本貼在腳本中)*
  <br>如果在 **Request Body** 多添加參數 **`, "html": true`**，可以直接返回 `HTML` 代碼文本

  ```bash
  curl -sX POST http://172.17.1.243:7860/ai/copilot \
    -H 'Content-Type: application/json'\
    -d '{ "lang": "Python", "prompt": "寫一個程序執行命令 ipmitool raw 6 1 判斷 00 在返回值中打印 Pass 不在就打印 Fail" }' | jq -r .response | tee test.py
  ```

  ```bash
  {
      "response": "\n\nimport os\nimport sys\n\n\ndef run_cmd(cmd):\n    p = os.popen(cmd)\n    return p.read()\n\n\ndef get_status(cmd):\n    p = os.popen(cmd)\n    return p.read()\n\n\ndef get_status_code(cmd):\n    p = os.popen(cmd)\n    return p.read()\n\n\nif __name__ == \"__main__\":\n    cmd = \"ipmitool raw 6 1\"\n    cmd_status = \"ipmitool raw 6 1 | grep '00'\"\n    cmd_status_code = \"ipmitool raw 6 1 | grep '00' | wc -l\"\n\n    print(run_cmd(cmd))\n    print(get_status(cmd_status))\n    print(get_status_code(cmd_status_code))\n\n    if \"00\" in get_status(cmd_status):\n        print(\"Pass\")\n    else:\n        print(\"Fail\")\n\n\n\"\"\"\nipmitool raw 6 1 | grep '00'\nipmitool raw 6 1 | grep '00' | w\n\"\"\"",
      "lang": "Python",
      "elapsed_time": 22.534089,
      "datetime": "2024-06-07 05:45:26"
  }
  ```

- 可以正常執行 AI 生成的 `Python` 腳本並符合預期！

  ```bash
  root@debian:~# python test.py
  20 00 02 12 02 8d dd b3 00 18 00 00 00 00 00

  20 00 02 12 02 8d dd b3 00 18 00 00 00 00 00

  1

  Pass
  ```

  ![api_demo](https://github.com/AllennLiu/fastapi-code-aicomplete/assets/27174570/752d6d17-47a8-4c89-b31b-b03c962703fe)

---

## 😵 Troubleshooting

- 如何監控 GPU 利用率？本地請先安裝 `cuda-toolkit` 工具，容器則會自行偵測。

  ```bash
  watch nvidia-smi
  ```

  ```bash
  Every 2.0s: nvidia-smi                                                                                                                                                                                                                                   Blade-Allen: Fri Jun  7 21:06:49 2024

  Fri Jun  7 21:06:49 2024
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 550.69                 Driver Version: 551.95         CUDA Version: 12.4     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA GeForce RTX 4070 ...    On  |   00000000:01:00.0 Off |                  N/A |
  | N/A   47C    P0             20W /  105W |       0MiB /   8188MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |  No running processes found                                                             |
  +-----------------------------------------------------------------------------------------+
  ```

---

## 🗂 參考來源

- https://github.com/THUDM/CodeGeeX2
- https://github.com/li-plus/chatglm.cpp
- https://github.com/THUDM/ChatGLM3/tree/main
