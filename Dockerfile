FROM hectorandac/yolov6_tx2_base:latest

RUN apt-get update
RUN apt-get upgrade -y

# Remove this and add to base later!
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install fvcore

COPY . .