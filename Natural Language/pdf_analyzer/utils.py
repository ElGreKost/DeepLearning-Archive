from langchain_openai import OpenAI
from pypdf import PdfReader
import pandas as pd
import re
import replicate
from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_doc):
    text = ''
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extracted_data(pages_data):
    template = """Extract all the following values : invoice no. Description, Quantinty, data, Unit price, Amount, Total
    email, phone number and address from this data: {pages}
    
    Expected output: remove any dollar symbols {{'Invoice no.': '1001329', 'Description': 'Office Chair', 'Quantity': '2', 'Date': '5/4/2018', 'Unit price': '100', 'Amount': '20', 'Total': '12020' and so on}}"""

    prompt_template = PromptTemplate(input_variables=["pages"], template=template)

    llm = OpenAI(temperature=.7)
    full_response = llm(prompt_template.format(pages=pages_data))

    # input_prompt = {
    #     "top_p": .9,
    #     "prompt": prompt_template.format(pages=pages_data),
    #     "temperature": .1,
    #     "max_new_tokens": 500
    # }
    # full_response = ''
    # for event in replicate.stream("meta/llama-2-70b-chat", input=input_prompt):
    #     full_response += event

    return full_response


def create_docs(user_pdf_list):
    """Iterate over files in that user uploaded PDF files, one by one"""
    print("Started creating docs")
    df = pd.DataFrame({"Invoice no.": pd.Series(dtype='str'),
                       "Description": pd.Series(dtype='str'),
                       "Quantinty": pd.Series(dtype='str'),
                       "Date": pd.Series(dtype='str'),
                       "Unit price": pd.Series(dtype='str'),
                       "Amount": pd.Series(dtype='int'),
                       "Total": pd.Series(dtype='str'),
                       "Email": pd.Series(dtype='str'),
                       "Phone number": pd.Series(dtype='str'),
                       "Address": pd.Series(dtype='str')})

    for filename in user_pdf_list:
        print('filename is {}'.format(filename))
        raw_data = get_pdf_text(filename)
        print('raw_data is {}'.format(raw_data))

        llm_extracted_data = extracted_data(raw_data)
        print('llm_extracted_data is {}'.format(llm_extracted_data))

        # Capturing just the required data and excluding andy unwanted test from the LLM response
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            data_dict = eval('{' + extracted_text + '}')
            print('data_dict is {}'.format(data_dict))
        else:
            print("No match found.")
        new_row = pd.DataFrame([data_dict])  # Creating a DataFrame from the dictionary
        df = pd.concat([df, new_row], ignore_index=True)  # Concatenating the new row
        print('*' * 10 + 'DONE' + '*' * 10)

    df.head()
    return df
