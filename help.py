import os
from pylatexenc.latex2text import LatexNodes2Text

def convert_latex_to_text(latex_content):
    """
    Convert LaTeX content to plain text.
    """
    return LatexNodes2Text().latex_to_text(latex_content)

def process_tex_files(source_folder, dest_folder):
    """
    Process .tex files in the source folder, parse them using Pylatexenc, and write the plain text
    to .txt files in the destination folder.
    """
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.tex'):
            source_file_path = os.path.join(source_folder, filename)
            dest_file_path = os.path.join(dest_folder, os.path.splitext(filename)[0] + '.txt')
            
            # Read the LaTeX content
            with open(source_file_path, 'r', encoding='utf-8') as source_file:
                latex_content = source_file.read()
            
            # Convert LaTeX to plain text
            plain_text = convert_latex_to_text(latex_content)
            
            # Write the plain text to the destination file
            with open(dest_file_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write(plain_text)
            
            print(f"Processed {filename} and saved to {os.path.basename(dest_file_path)}")

# Define the source and destination folders
source_folder = 'source'
dest_folder = 'attention'

# Process the .tex files
process_tex_files(source_folder, dest_folder)
