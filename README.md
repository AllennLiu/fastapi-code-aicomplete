# AI FastAPI CodeGeeX2

使用 `FastAPI` 搭配預訓練 `CodeGeeX2` 模型，通過 AI 智能生成代碼

> 目前測試起來，使用英文自然語言來描述代碼表現比較良好，用中文會給你寫一些比較奇怪甚至不相關的功能出來。
> 總的來說，**功能還是很強大**，目前正在整合 `CodeGeeX2-6B` 模型 + `ChatGLM3-6B` 模型，使兩個偉大的作品很好的被玩轉！

建置 **Docker** 鏡像來打包整個 **API** 服務成一個程序 *(包含模型文件大小 > `24G` )*

---

## 項目啟動

- 啟動 **API** 服務，這裡啟用了 **GPU Driver** 請確保安裝 Docker 的系統環境已安裝 `cuda-toolkit`，如果用 CPU 請刪除 `--gpus all` 與 `--gpu` 兩個參數。

  ```bash
  docker run -tid -p 7861:22 -p 7860:7860 \
    --gpus all \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    --ulimit nofile=65535 \
    --name copilot ai-fastapi-codegeex2/copilot:base bash service.sh --gpu
  ```

- 生成第一個 `Python` 腳本：

  ```bash
  curl -X POST http://127.0.0.1:7860/copilot \
    -H "Content-Type: application/json" \
    -d '{ "lang": "Python", "prompt": "# Write a quick sort function" }'
  ```

  ```bash
  {
      "response": " that takes in a list of numbers and\n\n\ndef quick_sort(lst):\n    if len(lst) <= 1:\n        return lst\n    pivot = lst[0]\n    less = [i for i in lst[1:] if i <= pivot]\n    greater = [i for i in lst[1:] if i > pivot]\n    return quick_sort(less) + [pivot] + quick_sort(greater)\n\n\nprint quick_sort([5, 3, 6, 2, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
      "lang": "Python",
      "elapsed_time": 24.878281,
      "datetime": "2024-06-06 11:57:51"
  }
  ```

  ![alt text](image.png)

---

## 參考來源

- https://github.com/THUDM/CodeGeeX2
- https://github.com/li-plus/chatglm.cpp
- https://github.com/THUDM/ChatGLM3/tree/main
