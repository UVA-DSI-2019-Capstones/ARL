#%%
from bs4 import BeautifulSoup as Soup, NavigableString, Tag
import os
import pandas as pd
from pathlib import Path

#%%
#HTML template for individual questions on AMT
html = """
<!-- Bootstrap v3.0.3 -->
<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
<div class="row col-xs-12 col-md-12"><!-- Instructions -->
<div class="panel panel-primary">
<div class="panel-heading"><strong>Instructions</strong></div>
<div class="panel-body">
<p><u><b>Context:</u></b> You are an American soldier who is meeting with the commander of a Chinese army platoon to discuss important business. <context></context></p>

<p><u><b>Intent:</u></b>  <intent></intent></p>

<p><u>Your instructions, given that context:</u> You are the American soldier. Please provide a response in your own words from the perspective of the American soldier that you feel matches the feedback above each text box.</p>

</div>
</div>
<!-- End Instructions --><!-- Content Body -->
<section>
<fieldset>
<div class="input-group"></div>
</fieldset>
<!-- End Content Body --></section>
</div>
</section>
<!-- close container -->
<style type="text/css">fieldset {
   padding: 10px;
   background:#fbfbfb;
   border-radius:5px;
   margin-bottom:5px;
}
</style>
"""
#%%
soup = Soup(html, features="lxml")

#Open the Excel file with the categories
dir = os.path.join(os.path.dirname(os.getcwd()), 'mechanical_turk', 'HTML_files')
file = os.path.join(dir, "context_categories.xlsx")
#Pick a sheet 
finaldf = pd.read_excel(open(file, 'rb'))

#%%
# #Only include resonses that the player can control
# newdf = df[(df.Speaker == 'Player') & (df.Identifier != 'no')]
# #Create a new dataframe with only the feedback, indentifier, and dialogue text
# finaldf = newdf.loc[:,['Dialogue Text', 'Feedback', 'Identifier']]
# #Reset indicies
# finaldf = finaldf.reset_index(drop=True)
# print(finaldf.head(20))
# dflen = finaldf.shape[0]

#Get dialogue from database
# file = "test_dialogue.xlsx"
# ctxt_df = pd.read_excel(open(file, 'rb'))


#
#%% Get context for game
file_1 = "Context_DME_edited.xlsx"
file = os.path.join(dir, file_1)
ctxt_df = pd.read_excel(open(file, 'rb'))
print(ctxt_df.head())

#%%
file_1 = "final context categories.xlsx"
file = os.path.join(dir, file_1)
description_df = pd.read_excel(open(file, 'rb'))
print(description_df.head())


#%%
#Get context for input
file_2 = "In-game context edited.xlsx"
file = os.path.join(dir, file_2)
intent_df = pd.read_excel(open(file, 'rb'))
print(intent_df.head())

#%%
os.chdir(os.path.join(dir, 'HTML_new_feedback_input'))

#%%
#Function for writing a label and inserting the text to the HTML file
def write_label(i, soup, rank):
    tag_name = 'label' + str(i)
    tag = soup.new_tag(tag_name)
    tag.string = description_df.loc[i,['Category Description']][0]
    in_tag = soup.new_tag("input")
    in_tag["class"] = "form-control"
    in_tag["name"] = str(rank)
    in_tag["size"] = "120"
    in_tag["type"] = "text"
    soup.find("div", {"class": "input-group"}).append(tag)
    soup.find("div", {"class": "input-group"}).append(in_tag)

#Loop over the dialogue entries
#%%
j = 1
unique_rank = set()
soup = Soup(html)
for i in range(len(description_df)):
    #Dialogue entries are formatted 1a-16c for DMC scenario, 
    #this try except handles converting these alphanumeric sequenes to ints
    try:
        cur_num = int(description_df.loc[i,['Section:']][0])
    except:
        cur_num = int(description_df.loc[i,['Section:']][0][0:1])
    cur_rank = description_df.loc[i,['Ranking:']][0]
    #If the number for the dialogue matches the number we are currently writing to the HTML file, the function is called
    if cur_num == j:
        print(j)
        print(' ')
        print('Unique rank ' + str(unique_rank))
        print('Current rank ' + str(cur_rank))
        if cur_rank not in unique_rank:
            write_label(i, soup, cur_rank)
            unique_rank.add(cur_rank)
    #If the number does not match, j is updated, and the function is called to write the first entry to the next HTML file
    else:
        ctxt = soup.find('context')
        ctxt_txt = ctxt_df[ctxt_df['Identifier'] == j]['Context'][j-1]
        ctxt.insert(0, NavigableString(ctxt_txt))
        
        intent = soup.find('intent')
        intent_txt = intent_df[intent_df['Section:'] == j]['On-screen context:'][j-1]
        intent.insert(0, NavigableString(intent_txt))

        file = "new_feedback_" + str(j) + ".html"
        j  += 1
        html_test = open(file, "w")
        html_test.write(str(soup))
        html_test.close()
        
        soup = Soup(html)
        if cur_rank not in unique_rank:
            write_label(i, soup, cur_rank)
        unique_rank = set()

j=19

#Else statement will not be called when i==16, so the final HTML file must be written outside of the loop
file = "new_feedback_" + str(j) + ".html"
html_test = open(file, "w")
html_test.write(str(soup))
html_test.close()
soup = Soup(html)
#%%
