#!/bin/bash

ps aux | grep kartonoe

rm -rf *-plots.png

python3 -u src/main.py >> logs.txt
mail -s "Training Model Done" e_kartono@hotmail.ca
