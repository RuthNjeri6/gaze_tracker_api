FROM python:3.9-slim
# setup environment variable
#RUN python3 -m venv /opt/venv
#ENV PATH="/opt/venv/bin:$PATH"

ADD requirements.txt /api/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /api/requirements.txt  --upgrade

# FROM python:3.9-alpine AS runtime-image
# COPY --from=compile-image /opt/venv opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /api
ADD . /api

EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]



# FROM python:3.9-alpine AS compile-image
# # setup environment variable
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # install dependencies 
# COPY ./requirements.txt ./requirements.txt
# RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt  --upgrade

# FROM python:3.9-alpine AS runtime-image
# COPY --from=compile-image /opt/venv opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# RUN python-on-whales download-cli
# # Run the application 
# WORKDIR /home/app
# COPY ./main.py .
# COPY ./model.py .
# CMD python main.py