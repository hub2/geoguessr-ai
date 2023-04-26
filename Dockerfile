FROM runpod/pytorch

RUN apt-get update && apt install -y vim htop
RUN mkdir -p /workspace/
RUN cd /workspace && git clone https://github.com/hub2/geoguessr-ai && cd geoguessr-ai && pip3 install -r requirements.txt 


RUN mkdir -p /workspace/geoguessr-ai/download_panoramas/downloads
