#!/bin/bash
fuser -k 5000/tcp
cd ..
cd Website/backend
node index.js
