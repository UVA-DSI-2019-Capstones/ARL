from bs4 import BeautifulSoup as Soup, NavigableString, Tag
import os
import pandas as pd
from pathlib import Path


#HTML template for individual questions on AMT
html = """
<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<!-- For the full list of available Crowd HTML Elements and their input/output documentation,
      please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

<!-- You must include crowd-form so that your task submits answers to MTurk -->
<crowd-form answer-format="flatten-objects">

    <!-- The crowd-classifier element will create a tool for the Worker to select the
           correct answer to your question -->
    <crowd-classifier 
      name="classify"
      header="Which category best describes the text?"
    >

      <classification-target>
      </classification-target>

     <!-- Use the short-instructions section for quick instructions that the Worker
            will see while working on the task. Including some basic examples of 
            good and bad answers here can help get good results. You can include 
            any HTML here. -->
      <short-instructions>
       <intro>The text is from an American army officer talking to Chinese army officer.</intro></p>
      <context>
      </context>
      </short-instructions></p>

      <

      <!-- Use the full-instructions section for more detailed instructions that the 
            Worker can open while working on the task. Including more detailed 
            instructions and additional examples of good and bad answers here can
            help get good results. You can include any HTML here. -->
      <full-instructions header="Text categorization instructions">
        <p><Other>Other</Other>: when the text cannot be understood (i.e. nonsensical answer) or none of the categories match at all</p>
        <p></p>
      </full-instructions>

    </crowd-classifier>
</crowd-form> 
"""

soup = Soup(html, features="lxml")

#Open the Excel file with the categories
file = "context_categories.xlsx"
#Pick a sheet 
finaldf = pd.read_excel(open(file, 'rb'))
# #Only include resonses that the player can control
# newdf = df[(df.Speaker == 'Player') & (df.Identifier != 'no')]
# #Create a new dataframe with only the feedback, indentifier, and dialogue text
# finaldf = newdf.loc[:,['Dialogue Text', 'Feedback', 'Identifier']]
# #Reset indicies
# finaldf = finaldf.reset_index(drop=True)
# print(finaldf.head(20))
# dflen = finaldf.shape[0]

#Get dialogue from database
file = "test_dialogue.xlsx"
ctxt_df = pd.read_excel(open(file, 'rb'))

#Get context 
file = "In-game context.xlsx"
context_df = pd.read_excel(open(file, 'rb'))

dir = Path(os.getcwd())
os.chdir(dir / 'HTML_category')


#Function for writing a label and inserting the text to the HTML file
def write_label(cat, soup):
    soup.find('crowd-classifier')['categories'] = str(cat)

#Loop over the dialogue entries
for i in range(0,len(ctxt_df)):
    dialogue = ctxt_df.loc[i,['Dialogue:']][0]
    soup.find('classification-target').append(dialogue)
    cur_num = int(ctxt_df.loc[i,['Section:']])
    context_text = context_df.loc[cur_num-1, ["On-screen context:"]][0]
    contxt = "Context given to American: " + context_text + '\n'
    soup.find("context").append(contxt)
    #If the number for the dialogue matches the number we are currently writing to the HTML file, the function is called
    print(cur_num)
    df = finaldf[finaldf['Section:'] == cur_num]
    df = df.reset_index(drop=True)
    print(df.head())
    cat = []
    for k in range (0,len(df)):
        print(df.loc[k, ['Category Description']][0])
        cat.append(df.loc[k, ['Category Description']][0])
    # Add option for other
    cat.append("Other")
    soup.find('crowd-classifier')['categories'] = str(cat)
    
    file = "User_input_" + str(i) + ".html"
    html_test = open(file, "w") 
    html_test.write(str(soup))
    html_test.close()
    soup = Soup(html, features="lxml")