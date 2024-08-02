FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV TZ="Asia/Taipei"

ENV NODE_SOURCE /workspace

COPY . $NODE_SOURCE

WORKDIR $NODE_SOURCE

RUN apt update -y && apt install -yq gcc g++ make cmake vim curl git jq wget ssh sshpass plink telnet unzip tmux

RUN date && conda update --all

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt || true

RUN echo 'alias vi="vim"' >> ~/.bashrc

EXPOSE 7860 22

USER root

ENV PYTHONIOENCODING  utf-8
ENV LANG              zh_CN.utf-8
ENV LANGUAGE          zh_CN:zh:en_US:en
ENV LOAD_MODEL_DEVICE cpu

CMD ["bash", "service.sh", "--prod", "--chat", "--code", "--multi-modal"]
