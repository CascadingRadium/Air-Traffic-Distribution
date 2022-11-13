#!/bin/bash
fuser -k 3000/tcp
cd ..
cd Website
npm start
