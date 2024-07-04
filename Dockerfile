# start by pulling the python image
FROM python:3-slim

# copy the requirements file into the image
COPY ./requirements.txt /myflaskproject/requirements.txt

# switch working directory
WORKDIR /myflaskproject

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y tdsodbc unixodbc-dev \
 && apt-get clean -y

RUN echo "[FreeTDS]\n\
Description = FreeTDS unixODBC Driver\n\
Driver = /usr/lib/arm-linux-gnueabi/odbc/libtdsodbc.so\n\
Setup = /usr/lib/arm-linux-gnueabi/odbc/libtdsS.so" >> /etc/odbcinst.ini

RUN pip install pyodbc
# copy every content from the local file to the image
COPY . /myflaskproject

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py"]