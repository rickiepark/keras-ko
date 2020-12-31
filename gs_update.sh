#!/bin/bash

PATH="$PATH":/Users/haesun/google-cloud-sdk/bin/
cd /Users/haesun/github/keras/keras-ko

source .env/bin/activate
git pull
cd scripts
python autogen.py make

gsutil -m rsync -r ../site gs://keras-ko.kr
