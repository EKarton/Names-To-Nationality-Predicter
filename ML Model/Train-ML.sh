#!/bin/bash

ps aux | grep kartonoe
kill -9 $(ps -aux | grep kartonoe | grep python | cut -d ' ' -f 2)

rm -rf logs.txt
rm -rf *-plots.png

python3 -u src/main.py > logs.txt
mail -s "Training Model Done" e_kartono@hotmail.ca
