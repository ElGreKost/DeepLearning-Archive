import streamlit as st
from dotenv import load_dotenv

from utils import *


def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extractor")
    st.title("Invoice Extractor Bot...")
    st.subheader("I can help you in extracting invoice data")

    # Upload the Invoices (pdf files)
    pdf = st.file_uploader("Upload invoices here", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Extract Data")

    if submit:
        with st.spinner("Loading invoices..."):
            df = create_docs(pdf)
            # st.write('managed to extract data from pdf')
            # st.write(df)
            st.write(df.head())

            # Saving the dataframe as CSV file
            data_as_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data_as_csv,
                "invoice_extracted_data.csv",
                "text/csv",
                key="download-tools-csv",
            )

        st.success("Hope I was able to save your time <3")


if __name__ == '__main__':
    main()
