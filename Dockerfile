FROM hectorandac/yolov6_tx2_base:latest

RUN apt-get update
RUN apt-get upgrade -y

COPY . .