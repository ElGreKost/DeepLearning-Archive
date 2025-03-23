# import pandas as pd
# import json
#
#
# # Function to parse the LLM output and create a DataFrame
# def process_llm_data(llm_data):
#     # Assuming llm_data is a string that exactly matches the dictionary format
#     # Safely convert string representation of dictionary to a dictionary
#     data_dict = eval(llm_data.strip())
#
#     # Convert the dictionary to a DataFrame
#     # Since our example has comma-separated values in strings for some fields, let's handle that
#     for key, value in data_dict.items():
#         if ',' in value:
#             data_dict[key] = [v.strip() for v in value.split(',')]
#         else:
#             data_dict[key] = [value]
#
#     df = pd.DataFrame(data_dict)
#
#     # Saving the DataFrame to a CSV file
#     csv_file_path = "Invoice_Data.csv"
#     df.to_csv(csv_file_path, index=False)
#
#     return csv_file_path
#
#
# # Example usage
# llm_extracted_data = """
# {
#     'Invoice no.': '001',
#     'Description': 'Widget A, Widget B',
#     'Quantity': '2, 5',
#     'Date': '2024-04-13',
#     'Unit price': '$30.00, $20.00',
#     'Amount': '$60.00, $100.00',
#     'Total': '$160.00',
#     'Email': 'info@gptwidgets.com',
#     'Phone number': '(123) 456-7890',
#     'Address': '123 AI Lane, Model Town, OpenAI'
# }
# """
#
# # Call the function to process the data and create a CSV
# csv_path = process_llm_data(llm_extracted_data)
# print(f"Data saved to CSV at: {csv_path}")

import pandas as pd
from utils import get_pdf_text, extracted_data
from dotenv import load_dotenv

load_dotenv()


def save_data_to_csv(data, filename):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    print(df)
    # Save the DataFrame to a CSV file
    csv_path = f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


# Example usage
print('filename is {}'.format("Invoice_001.pdf"))
raw_data = get_pdf_text("Invoice_001.pdf")
print('raw_data is {}'.format(raw_data))

llm_extracted_data = extracted_data(raw_data)
print('llm_extracted_data is {}'.format(llm_extracted_data))

# Assuming llm_extracted_data is a dictionary containing the extracted fields
save_data_to_csv(llm_extracted_data, "Extracted_Invoice1")
