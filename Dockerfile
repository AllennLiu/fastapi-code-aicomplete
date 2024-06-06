FROM code-geex-2:latest

ENV TZ="Asia/Taipei"

ENV NODE_SOURCE /workspace

COPY . $NODE_SOURCE

WORKDIR $NODE_SOURCE

RUN date && apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ make vim curl git \
        ssh sshpass plink telnet unzip

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || true

RUN echo 'alias vi="vim"' >> ~/.bashrc

EXPOSE 7860 22

USER root

ENV PYTHONIOENCODING utf-8

CMD ["bash", "service.sh"]
