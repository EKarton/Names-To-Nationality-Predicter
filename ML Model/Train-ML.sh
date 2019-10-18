#!/bin/bash

python3 -u src/main.py > logs.txt
mail -a *-plots.png -s "Training Model Done" e_kartono@hotmail.ca
