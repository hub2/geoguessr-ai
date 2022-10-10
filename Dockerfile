FROM runpod/pytorch


#RUN apt-get -y update && apt-get install git

RUN pip install geopandas pygeohash
RUN mkdir -p /app
RUN cd /app && git clone https://github.com/hub2/geoguessr-ai
