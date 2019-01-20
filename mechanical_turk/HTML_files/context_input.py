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

    <crowd-instructions link-text="View instructions" link-type="button">
        <short-summary>
            <p>Collect utterances for intent</p>
        </short-summary>

        <detailed-instructions>
            <h3>Collect utterances for intent</h3>
            <p>
              Given a context and an intent, write how you would express the intent using natural language. 
              Simply write what you would say if you were in the given situation.
        </p>
        </detailed-instructions>

        <positive-example>
            <h3>Context</h3>
            <p>You bought a pair of shoes online but they don't fit</p>

            <h3>Intent</h3>
            <p>You want to try to return the shoes via an online customer service chat bot</p>

            <h3>Response</h3>
            <p>I would like to return a pair of shoes</p>
        </positive-example>


        <negative-example>
            <h3>Context</h3>
            <p>You bought a pair of shoes online but they don't fit</p>

            <h3>Intent</h3>
            <p>You want to try to return the shoes via an online customer service chat bot</p>

            <h3>Response</h3>
            <p>Don't fit</p>
         </negative-example>
    </crowd-instructions>

    <p>Write what you would say in the given situation:</p>

    <!-- Your contexts and intents will be substituted for the ${context} and ${intent} variables when you 
           publish a batch with an input file containing multiple contexts and intents -->
    <p><context>Context: </context></p>
    <p><intent>Intent: </intent></p>

    <crowd-input name="utterance" placeholder="Type what you would say here..." required></crowd-input>
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
# file = "test_dialogue.xlsx"
# ctxt_df = pd.read_excel(open(file, 'rb'))


#Get context for game
file_1 = "Context_DME_edited.xlsx"
context_df = pd.read_excel(open(file_1, 'rb'))
print(context_df.head())
#Get context for input
file_2 = "In-game context edited.xlsx"
intent_df = pd.read_excel(open(file_2, 'rb'))
print(intent_df.head())

dir = Path(os.getcwd())
os.chdir(dir / 'HTML_context_input')


#Function for writing a label and inserting the text to the HTML file
def write_label(cat, soup):
    soup.find('crowd-classifier')['categories'] = str(cat)

#Loop over the dialogue entries
for i in range(0,len(context_df)):
    intent = intent_df.loc[i,['On-screen context:']][0]
    soup.find('intent').append(intent)
    context = context_df.loc[i,['Context']][0]
    soup.find('context').append(context)
    file = "Context_input_" + str(i+1) + ".html"
    html_test = open(file, "w") 
    html_test.write(str(soup))
    html_test.close()
    soup = Soup(html, features="lxml")