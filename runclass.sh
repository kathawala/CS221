#!/bin/bash

python2 classifier2.py

#/g' at the end is global substitution
sed -i 's/numbers=False/numbers=True/' classifier2.py

python2 classifier2.py

sed -i 's/numbers=True/numbers=False/' classifier2.py