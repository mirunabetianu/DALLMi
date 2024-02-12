#!/bin/bash

for removal_type in "label" 
do
   for seed in 42
   do
    /home/mbetianu/venv/bin/python /home/mbetianu/pu_bert/pu_bert_trainer.py --removal_strategy $removal_type --seed $seed
   done
done