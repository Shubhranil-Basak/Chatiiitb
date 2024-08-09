# ChatIIITB

ChatIIITB is an tool that uses RAG to read documents and answers questions based on those documents. It uses publically available documents and informations on the IIITB [website](https://iiitb.ac.in).

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Technology Stack](#technology-stack)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
    - [APIs Used](#apis-used)
    - [Application Structure](#application-structure)
- [Challenges](#challenges)
- [Future Scope](#future-scope)
- [Contributions](#contributions)
- [License](#license)
- [References](#references)

## Problem Statement
A rag system based on IIITB docs, capable of answering questions from the docs. For example:
- "❓ My fees have not been refunded, whom should I talk to?"
- "❓ A company has offered me a job, whom should I inform?"

## Objectives
- Develop a tool that can retrieve information from IIITB documents with high accuracy.
- Enable users to get precise answers to their queries based on the content of IIITB documents.
- Ensure the system is user-friendly and accessible for all IIITB students and staff.

## Technology Stack

### Frontend
- **streamlit**: For the user interface.

### Backend
- **sqlite3**: For storing the vector tables.

### APIs and libraried
- **Gemini API**: For generating sentences from the retrieved embeddings.
- **LangChain**: For RAG functionalities.

### Database
- **ChromaDB**: For document storage and retrieval.

## Key Features
- Document Upload: Allows admins to upload IIITB documents to be indexed.
- Question Answering: Users can ask questions, and the system will provide answers based on the uploaded documents.
- Dynamic Document Indexing: Updates the document database dynamically as new documents are added.
- Interactive Interface: User-friendly interface for easy interaction.

## Getting Started

To get started with ChatIIITB, follow these steps:

1. Clone the repository using the following command:
```bash
git clone https://github.com/Shubhranil-Basak/Chatiiitb.git
```
2. Install the dependencies using the following command:
```bash
cd Chatiiitb
```
```bash
pip install -r requirements.txt
```
3. Set up environment variables by making a `.env` file and filling the following information:
Get your Gemini API Key here: https://aistudio.google.com/app/apikey
You need to enable your youtube API key in the google developer console and copy it from there.
```plaintext
GOOGLE_API_KEY = YOUR_GEMINI_API
```
4. to use the chatbot, run the following command:
```bash
streamlit run chat.py
```
## Usage
1. There are pre-loaded files from the IIITB website in the `data` directory.
2. To add new data, add the data in the `data` directory and clear the `local_chroma_db` directory.
3. Run the chatbot using the streamlit command and wait for some time.
4. after opening the chat interface, write the quesion and press the `send` button.

## Implementation Details

### APIs Used
1. **Gemini API**: This is used to generate the statements from the given context from the documents.

### Project Structure
1. `data` folder stores the relavent documents in either pdf or markdown format.
2. `local_chroma_db` folder stores the vector embeddings for the chunks of informations retrieved from the documents.
3. `chat.py` is the main file which creates the vector embeddings and makes API calls to Gemini.

![Chatiiitb drawio](https://github.com/user-attachments/assets/4e7bb845-0fe2-46f2-9c06-98a6db4ed64c)


## Challenges
- Ensuring accurate and relevant document retrieval for varied queries.
- Handling document format inconsistencies and ensuring smooth indexing.
- Optimizing the system for fast response times.
- Finding good hyperparameters like `chunk_overlap`, `length_function`, `chunk_size`, etc.

## Future Scope
- **Expanded Document Types**: Support for a wider range of document types and formats.
- **Enhanced Interaction**: More interactive features such as highlighting relevant document sections in responses.
- **Integration with IIITB Systems**: Seamless integration with existing IIITB information systems for real-time data access.
- **Dynamic Add, Delete feature**: Adding feature to create vector embeddings for newly added or deleted documents on the go without erasing the exsisting vector embedding table.
- **Using MOE architecture**: Using the State of the Art MOE architecture with different APIs like OpenAI API and sonnet to generate better responses.

## Contributions

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## References
- IIITB Official Website: https://iiitb.ac.in
- Gemini API Documentation: https://ai.google.dev/gemini-api/docs
- LangChain Documentation: https://python.langchain.com/v0.2/docs/introduction/
