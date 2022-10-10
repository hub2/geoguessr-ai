FROM runpod/pytorch


#RUN apt-get -y update && apt-get install git

RUN mkdir -p /workspace
RUN cd /workspace && git clone https://github.com/hub2/geoguessr-ai && cd geoguessr-ai && pip install -r requirements.txt
