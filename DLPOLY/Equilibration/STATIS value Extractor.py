"""
Author @ Eric(MRES 2026) 
This is my attempt at extracting data from statis files 

""" 
import tkinter,tkinter.filedialog #data selection # used for file selection(n.b. will open in a separate window)
import pandas as pd #file reacting and data extraction
import re # used for file handling / reading
import matplotlib.pyplot as plt # for plotting the data
from matplotlib.ticker import MaxNLocator # used to make the plots less messy


####### USER INPUT SECTION ########

filename = tkinter.filedialog.askopenfilename()
tkinter.Tk().withdraw() # this code allows us to select files
#this saves headaches when inputting filepaths 

f = open(filename,"r") # read the file

rows = f.read() # plug each row into a long str

f.close() # close the file 



########### Reading data and Reading Number of Steps ##############################

#Obtaining our Values and Steps from the str



pattern_2 = re.compile(r'^[ \t]*\d+[ \t]+[+-]?\d+\.\d+E[+-]\d+[ \t]+\d+[ \t]*$', re.MULTILINE)
#the above pattern identifies the step label 
chunked_steps = pattern_2.findall(rows) # this uses pattern to identify the steps



######### INSERT USER DEFINED FUNCTIONS BELOW #####################################

# #Producing a list of lists with each term separated into string
def str_splitter(chunked_list):  

 """
 This is a function designed to take our regex processed lists 
 and break them down into individual values. 
    
 index:
 chunked_list = the list of regex processed values
 """ 
 unchunked_list = [] 
    
 for row in chunked_list:
     
    x = row.split() # str method to break apart "value blocks" into single values
    unchunked_list.append(x) 
 
 return unchunked_list
        

def value_steps_dataframe(value):
    """
    This is a function which finds the value at a given stats steps 
    This converts these values into a df which can be plugged into a matplotlib
       
    value = the index of the parameter in the list you are looking for (index)
    
    This function expects three terms in record i, (see dl_poly classic manual,STATIS section)    
    """
    #Producing empty lists to load our step number and value into
    col_steps = []
    col_value = []

    for x in unchunked_steps:
        col_steps.append(x[0])
        
        
    for x in unchunked_values : 
        col_value.append(x[value])
        
    df = pd.DataFrame({'steps':col_steps,'value':col_value})
    
    
    
    #The section below exists as the regex returns a str which caused bugs below fixed 04/11/25
    df['steps'] = pd.to_numeric(df['steps'], errors="coerce")
    df['value'] = pd.to_numeric(df['value'], errors="coerce")
    
    
    return df
            
def lazy_plot(df, name):
    """
    df  = the dataframe produced by value_steps_dataframe
    name = the title of the plot being produced
    
    """
    
    
    fig, ax = plt.subplots()
    

    ax.scatter(df["steps"], df["value"])

    ax.set_title(name)
    
    ax.yaxis.set_major_locator(MaxNLocator(15))
    ax.xaxis.set_major_locator(MaxNLocator(5))

    plt.show()




####################### Proccessing using defined functions ###############################


unchunked_steps = str_splitter(chunked_steps) # "splitting step-time-number of values into individ values  "

number_of_vals = int(unchunked_steps[0][2])


pattern = re.compile(rf'(?:[+-]?\d+\.\d+E[+-]\d+\s+){{{number_of_vals-1}}}[+-]?\d+\.\d+E[+-]\d+') 
#the above pattern identifies all of the values in ONE step
chunked_values = re.findall(pattern,rows) # this uses the pattern to identify all of values

unchunked_values = str_splitter(chunked_values) # spliting "value blocks into single vals"


#####INSERT SYSTEM RELEVANT VALUES BELOW#####

"""
TO ADD TO THIS SECTION - 

1) go to dl poly manual 
    output files>statis

2) find the index of the value that you want

then:
    
#LABEL FOR INFORMATION

name_of_variable  = value_steps_dataframe([INSERT INDEX OF VARIABLE HERE])

lazy_plot(name_of_variable, "title for the outputted plot")


"""


#Plot of E for equilibration

system_E = value_steps_dataframe(0)

lazy_plot(system_E,"system energy")

#Plot for system T

system_T = value_steps_dataframe(1)

lazy_plot(system_T,"system temp")


#This section is for things relevant to a NPT simulation

if number_of_vals > 38:
    
    
    
    
    
    #### Unit Cell Volume Change 
    
    unit_cell_change = value_steps_dataframe(18)
 
    lazy_plot(unit_cell_change,"System Volume")





############# Stats Bit #################

stats = []

stats.append("systemE")
stats.append(system_E["value"].describe())

stats.append("systemT")
stats.append(system_T["value"].describe())

stats.append("systemV")
stats.append(unit_cell_change["value"].describe())



