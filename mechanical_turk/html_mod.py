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
<p>Example instructions: Please insert an alternative phrasing of the following prompts:</p>
<ul>
    <li>Do not change the meaning of the prompt.</li>
    <li>Try to match the tone of the prompt.</li>
    <li>Check spelling and grammar</li>
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
df = pd.read_excel(open(file, 'rb'), sheet_name = 'DME')
#Only include resonses that the player can control
newdf = df[(df.Speaker == 'Player') & (df.Identifier != 'no')]
#Create a new dataframe with only the feedback, indentifier, and dialogue text
finaldf = newdf.loc[:,['Dialogue Text', 'Feedback', 'Identifier']]
#Reset indicies
finaldf = finaldf.reset_index(drop=True)
print(finaldf.head(20))
dflen = finaldf.shape[0]

# dir = os.getcwd()
# os.chdir(dir + '\\html_files')

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
		j  += 1
		file = "test" + str(i) + ".html"
		html_test = open(file, "w")
		html_test.write(str(soup))
		html_test.close()
		soup = Soup(html)
		write_label(i, soup)

#Else statement will not be called when i==16, so the final HTML file must be written outside of the loop
file = "test" + str(i) + ".html"
html_test = open(file, "w")
html_test.write(str(soup))
html_test.close()
