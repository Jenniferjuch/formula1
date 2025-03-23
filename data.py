import numpy as np 
import pandas as pd 
import fastf1

event = fastf1.get_event(2024,15)
# print(event)

session = fastf1.get_session(2024,15,'R')
session.load()