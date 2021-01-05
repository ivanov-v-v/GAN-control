#!/bin/bash

IPYTHON_PORT=2020
jupyter notebook --port="${IPYTHON_PORT}" --port-retries=0 --ip='::' --no-browser

