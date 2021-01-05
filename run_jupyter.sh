#!/bin/bash

IPYTHON_PORT=280597
jupyter notebook --port="${IPYTHON_PORT}" --port-retries=0 --ip='::' --no-browser

