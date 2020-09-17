# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:30:19 2020

@author: rruiz26
"""

import gspread
import json

from oauth2client.service_account import ServiceAccountCredentials
import os
dir =  os.getcwd()
#from pprint import pprint  
with open("./db.json","r") as read_file:
    dict = json.load(read_file)

first_name = dict['users'][-1]['first name']
last_name = dict['users'][-1]['last name']
gender = dict['users'][-1]['gender']
user_id = dict['users'][-1]['messenger user id']    
firsthalf = user_id[0:8]
secondhalf = user_id[8:]

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(dir +"/python/creds.json", scope)

client = gspread.authorize(creds)

sheet = client.open("Testing").sheet1

insertRow = ["",first_name,last_name,gender,"This came from Heroku Server",firsthalf,secondhalf]

sheet.append_row(insertRow)


# todo for project 

#get this python code (with adaptive experminet code) to run when there is a post to the server 


#notes on zapier integration make column/row called newest subsscriber