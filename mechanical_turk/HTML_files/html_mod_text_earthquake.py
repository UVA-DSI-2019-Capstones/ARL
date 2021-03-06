from bs4 import BeautifulSoup as Soup, NavigableString, Tag
import os
import pandas as pd

#HTML template for individual questions on AMT
html = """
<!-- Bootstrap v3.0.3 -->
<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
<section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
<div class="row col-xs-12 col-md-12"><!-- Instructions -->
<div class="panel panel-primary">
<div class="panel-heading"><strong>Instructions</strong></div>
<div class="panel-body">
<p><u>Context:</u> <context></context></p>
<p><u>Your instructions, given that context:</u> Please re-word the following prompts in your own words. While doing so, please also:</p>
<ul>
    <li>Maintain the meaning of the prompt.</li>
    <li>Try to match the tone of the prompt.</li>
    <li>Check spelling and grammar.</li>
</ul>
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

soup = Soup(html)

#Open the Excel file with the dialogue 
file = "Dialogue.xlsx"
#Pick a sheet 
df = pd.read_excel(open(file, 'rb'), sheet_name = 'Earthquake')
#Only include resonses that the player can control
newdf = df[(df.Speaker == 'Player') & (df.Identifier != 'no')]
#Create a new dataframe with only the feedback, indentifier, and dialogue text
finaldf = newdf.loc[:,['Dialogue Text', 'Feedback', 'Identifier']]
#Reset indicies
finaldf = finaldf.reset_index(drop=True)
print(finaldf.head(20))
print(finaldf.loc[50,['Feedback']][0])
dflen = finaldf.shape[0]
print(dflen)

#Open the Excel with context
file = "Context_earthquake.xlsx"
ctxt_df = pd.read_excel(open(file, 'rb'))

dir = os.getcwd()
os.chdir(dir + '\\HTML_earthquake')

#Function for writing a label and inserting the text to the HTML file
def write_label(i, soup):
    tag_name = 'label' + str(i)
    tag = soup.new_tag(tag_name)
    tag.string = finaldf.loc[i,['Dialogue Text']][0]
    in_tag = soup.new_tag("input")
    in_tag["class"] = "form-control"
    in_tag["name"] = "Q5FreeTextInput"
    in_tag["size"] = "120"
    in_tag["type"] = "text"
    soup.find("div", {"class": "input-group"}).append(tag)
    soup.find("div", {"class": "input-group"}).append(in_tag)

#Loop over the dialogue entries
j = 1
for i in range(0,dflen):
    #Dialogue entries are formatted 1a-16c for DMC scenario, 
    #this try except handles converting these alphanumeric sequenes to ints
    try:
        cur_num = int(finaldf.loc[i,['Identifier']][0][0:2])
    except:
        cur_num = int(finaldf.loc[i,['Identifier']][0][0:1])
    #If the number for the dialogue matches the number we are currently writing to the HTML file, the function is called
    if cur_num == j:
        write_label(i, soup)
    #If the number does not match, j is updated, and the function is called to write the first entry to the next HTML file
    else:
        ctxt = soup.find('context')
        ctxt_txt = ctxt_df[ctxt_df['Identifier'] == j]['Context'][j-1]
        ctxt.insert(0, NavigableString(' ' + ctxt_txt))
        file = "earthquake_text" + str(j) + ".html"
        j  += 1
        html_test = open(file, "w")
        html_test.write(str(soup))
        html_test.close()
        soup = Soup(html)
        write_label(i, soup)

j=23

#Else statement will not be called at the end, so the final HTML file must be written outside of the loop
ctxt = soup.find('context')
ctxt_txt = ctxt_df[ctxt_df['Identifier'] == j]['Context'][j-1]
ctxt.insert(0, NavigableString(ctxt_txt))
file = "earthquake_text" + str(j) + ".html"
html_test = open(file, "w")
html_test.write(str(soup))
html_test.close()