FROM seven6306/pretrained-model:ai-fastapi-copilot

ENV TZ="Asia/Taipei"

ENV NODE_SOURCE /workspace

RUN rm -rf $NODE_SOURCE/*

COPY . $NODE_SOURCE

WORKDIR $NODE_SOURCE

RUN date && apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ make vim curl git jq \
        ssh sshpass plink telnet unzip tmux

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || true

RUN echo 'alias vi="vim"' >> ~/.bashrc

EXPOSE 7860 22

USER root

ENV PYTHONIOENCODING  utf-8
ENV LANG              zh_CN.utf-8
ENV LANGUAGE          zh_CN:zh:en_US:en
ENV LC_ALL            zh_CN.utf-8
ENV LOAD_MODEL_DEVICE cpu

CMD ["bash", "service.sh"]
