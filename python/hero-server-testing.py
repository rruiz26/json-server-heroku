# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:30:19 2020

@author: rruiz26
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
dir =  os.getcwd()
#from pprint import pprint  

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(dir +"/python/creds.json", scope)

client = gspread.authorize(creds)

sheet = client.open("Testing").sheet1

insertRow = ["hello","this","is", "from", "the", "Heroku", "server"]

sheet.insert_row(insertRow,1)


# todo for project 

#get this python code (with adaptive experminet code) to run when there is a post to the server 


#notes on zapier integration make column/row called newest subsscriber