#!/bin/bash

for i in 0 0.2 0.4 0.6 0.8 1 1.2 1.4
do
  python main.py -w $i
done