# FastAPI Code AI Complete

使用 `FastAPI` 搭配預訓練 `CodeGeeX2 + GLM-4` 模型，通過 `AI` 智能生成代碼

目前測試起來，**整體功能非常強大**，對於硬件環境沒那麼富裕的開發人員很友好 👀

目前已整合了 `CodeGeeX2-6B` + `GLM-4-9B-Chat` + `GLM-4v-9B` 量化模型，使這 3 個由**清华大学 KEG 智谱**偉大的作品很好的合併使用！

> **Docker** 鏡像來打包整個 **FastAPI** 服務成一個程序 *(包含模型文件大小 > `31G` )*
>
> 歡迎訪問我們的倉庫 👉 [https://hub.docker.com/r/seven6306/pretrained-model](https://hub.docker.com/r/seven6306/pretrained-model)
>
> 也誠摯歡迎加入我們一起玩玩有趣的技術 🤝

---

## 🎥 效果演示 *(Stream)*

### 👩‍💻 `AI` 依指定需求生成代碼 🔗 [http://127.0.0.1:7860/copilot/demo](http://127.0.0.1:7860/copilot/demo)

![coding](https://github.com/AllennLiu/fastapi-code-aicomplete/assets/27174570/a2b37382-2487-4f02-8b3d-3e2d505d7070)

### 🤖 與聊天機器人互動 🔗 [http://127.0.0.1:7860/chat/utils/demo](http://127.0.0.1:7860/chat/utils/demo)

![chatbot](https://github.com/AllennLiu/fastapi-code-aicomplete/assets/27174570/3fa83e23-ed30-4e25-9e02-4db364e48ebb)

---

## 🚀 快速開始

啟動 **Web API** 服務，這裡啟用了 **`GPU Driver`** 請確保安裝 Docker 的系統環境已安裝 `cuda-toolkit`。

> 如果用 **CPU** 請刪除 `--gpus all` 與 `-e "LOAD_MODEL_DEVICE=gpu"` 兩個參數

```bash
docker run -tid -p 7861:22 -p 7860:7860 \
  --gpus all \
  -e "LOAD_MODEL_DEVICE=gpu" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /root/.ssh:/root/.ssh:ro \
  --ulimit nofile=65535 --privileged=true --restart=always \
  --name copilot seven6306/pretrained-model:ai-fastapi-glm4
```

手動啟動 **Web** 服務腳本

```bash
root@bec85164e078:/workspace# bash service.sh -h
Usage: service.sh [Option] argv

FastAPI backend service manager.

Options:
    -s, --ssh            serve with SSH server only
    --kill, --terminate  terminate existing web process immediately
    --stag, --test       serve environ with deployment
    --prod, --main       serve environ with production
    --chat               enable chat model
    --code               enable code model
    --multi-modal        enable multi modal model

```

---

**在 `FastAPI` Docs 下嘗試 [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)**

### 👨‍💻 代碼生成

可以选择的参数

- `lang` - 程序的語言如 `Python, JavaScript, Shell, ...`
- `prompt` - 描述程序的需求
- `html` - 是否直接返回代碼，默認為 `false`

```bash
curl -sX POST http://127.0.0.1:7860/copilot/coding \
  -H 'Content-Type: application/json'\
  -d '{ "lang": "Python", "prompt": "寫一個程序執行命令 ipmitool raw 6 1 判斷 00 在返回值中打印 Pass 不在就打印 Fail", "html": true }' | python
```

#### 可以正常執行 **AI** 生成的 `Python` 腳本並符合預期！

```bash
20 00 02 12 02 8d dd b3 00 18 00 00 00 00 00

20 00 02 12 02 8d dd b3 00 18 00 00 00 00 00

1

Pass
```

![api_demo](https://github.com/AllennLiu/fastapi-code-aicomplete/assets/27174570/752d6d17-47a8-4c89-b31b-b03c962703fe)

---

### 💬 進行聊天

可以选择的参数

- `query` - 用戶輸入信息
- `history` - 進行**多輪式對話** *(數組形式 `[ dict... ]`)*
- `role` - 交互角色如 `user, assistant, system, observation`，默認為 `user`
- `html` - 是否直接返回機器人回覆的信息，默認為 `false`

```bash
curl -X POST http://127.0.0.1:7860/chat/conversation \
  -H 'Content-Type: application/json' -d '{ "query": "你最近還好嗎?" }'
```

```json
{
  "response": "我很好，谢谢。你呢？",
  "history": [
    {
      "role": "user",
      "content": "你最近還好嗎?"
    },
    {
      "role": "assistant",
      "metadata": "",
      "content": "我很好，谢谢。你呢？"
    }
  ],
  "datetime": "2024-06-14 05:40:02",
  "elapsed_time": 1191.721331
}
```

---

## 🌐 環境準備

- 本項目已經將已配置好的環境鏡像，推在 `Dockerhub` 上了，**大小 `40G`** *(內含 `CodeGeeX2-6B` + `GLM-4-9B-Chat` LLM)*
- 鏡像連結 🔗 [seven6306/pretrained-model:ai-fastapi-glm4](https://hub.docker.com/repository/docker/seven6306/pretrained-model/tags)
- 如果您要手動 Build **Docker** 鏡像 `ai-fastapi-glm4` *(使用 Docker 來避免環境依賴等問題)*
  1. 確認環境下有 `git-lfs` 工具，用 `Git` 來拉取 `hugging-face` 上的大型文件所需的工具

     ```bash
     git clone https://github.com/AllennLiu/fastapi-code-aicomplete.git
     cd fastapi-code-aicomplete
     git lfs install
     ```

  2. 下載兩個大模型 `CodeGeeX2-6B` 與 `GLM-4-9B-Chat` 至項目文件夾下 *(模型加起來有 `40G` 左右要下很久…)*

     ```bash
     git clone https://huggingface.co/THUDM/glm-4-9b-chat
     git clone https://huggingface.co/THUDM/codegeex2-6b
     ```

  3. Build 鏡像 - 來源鏡像：`pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel`

     ```bash
     docker build --no-cache -t seven6306/pretrained-model:ai-fastapi-glm4 . \
        --build-arg "HTTP_PROXY=http://admin:ZD7EdEpF9qCYpDpu@10.99.104.250:8081/" \
        --build-arg "HTTPS_PROXY=http://admin:ZD7EdEpF9qCYpDpu@10.99.104.250:8081/" \
        --build-arg "NO_PROXY=localhost,127.0.0.1,.example.com"
     # 掛代理 --build-arg 是我內部環境用的，不需要可以刪除
     ```

---

## 😵 Troubleshooting

如何監控 GPU 利用率？本地請先安裝 `cuda-toolkit` 工具，**Docker Container** `--gpus all` 會自動依賴本地的驅動。

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

## 💡 參考來源

- [THUDM/CodeGeeX2](https://github.com/THUDM/CodeGeeX2)
- [THUDM/GLM-4](https://github.com/THUDM/GLM-4)
- [chatglm.cpp](https://github.com/li-plus/chatglm.cpp)
