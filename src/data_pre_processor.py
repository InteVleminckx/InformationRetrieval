import ast
import os
import subprocess
import sys

import pandas as pd


def safe_eval(text):
    """
    This function evaluates an expression node or a string consisting of a Python literal or container display.
    https://www.educative.io/answers/what-is-astliteralevalnodeorstring-in-python
    :param text: input text to evaluate
    :return:
    """
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


class DataPreProcessor:

    def __init__(self, input_file, cwd):

        self.preprocessed_data = {}
        self.docs_s_tokenized = []
        self.docs_l_tokenized = []
        self.titles = []
        self.data_set = {}
        self.preprocess_done = False
        self.preprocess(input_file, 'data/lowered_input.csv', cwd)

    def extract_data(self, dataset, output_file):
        """
        Reads out the dataset correctly and lowers the text
        :param dataset: the dataset that needed to be read out
        :param output_file: the file where the data is going to be written in
        """

        # Read out data in a data frame
        df = pd.read_csv(dataset)

        # Convert titles to a list
        self.titles = df['Title'].to_list()

        # Do some cleanup
        for i, title in enumerate(self.titles):
            self.titles[i] = title.replace("&amp;", "&")

        # Read out the sections as lists
        df['Sections'] = df['Sections'].apply(safe_eval)

        # Reading out the section and store them per title
        with open(output_file, 'w') as output:
            for i, (title, sections) in df.iterrows():
                title = title.replace("&amp;", "&")
                self.data_set[title] = ''

                document = ""

                for i, section in enumerate(sections):
                    header, paragraph = section[0] if i > 0 else title, section[1]
                    self.data_set[title] += f"{header}\n{paragraph}"
                    document += paragraph.encode('utf-8', 'ignore').decode().lower()

                document = document.replace("\n", "")

                # Write the lowered document to the output file
                output.write(f'{title.lower()}, {document}\n')

    def preprocess(self, input_file, output_file, cwd):
        """
        Preprocesses the dataset
        :param input_file: the input file of the dataset
        :param output_file: the output file that will contain the preprocessed data
        :param cwd: the current working directory
        """

        # Extracting the data
        self.extract_data(input_file, output_file)

        output_file = f"{cwd}/{output_file}"
        with open(input_file, 'r') as input:
            with open(output_file, 'w') as output:
                for line in input.readlines():
                    # Already lowering the text, because this can be done faster here than in c++
                    line = line.lower()
                    line = line.replace("&amp;", "&")
                    output.write(line)

        new_input_file = output_file
        output_file = f"{cwd}/{'data/preprocessed.csv'}"

        # Create output file
        with open(output_file, "w") as f:
            f.write("")

        # Call the C++ program that will preprocess the data
        num_processes = 10
        exec_path = f"{cwd}/src/data_preprocessor/dpp"

        # Compile cpp file
        os.system(f"g++ -o {exec_path} {exec_path}.cpp")

        try:

            os.chdir(f"src/data_preprocessor")
            # Get return value of the cpp code
            result = subprocess.run(f"./dpp {new_input_file} {output_file} {num_processes}", stdout=subprocess.PIPE,
                                    shell=True, text=True)
            os.chdir("../..")

            # If the C++ program returns 1, something went wrong so stop the program
            if str(result.returncode) == "1":
                print("Error: there occurred an error during preprocessing the data in C++")
                sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)

        with open(output_file, 'rb') as f:
            for line in f.readlines():
                line = line.decode('utf-8', 'ignore')
                line = line.split(',')

                # If the title contains comma's, merge the title back to gather
                if len(line) > 2:
                    new_line = ["", ""]
                    for i, part in enumerate(line):
                        if i == len(line) - 1:
                            # Remove last added comma
                            new_line[0] = new_line[0][:-1]
                            new_line[1] = part
                        else:
                            new_line[0] += part + ","
                    line = new_line

                # Saving the preprocessed data as tokenized strings or lists
                self.docs_s_tokenized.append(line[1])
                self.docs_l_tokenized.append(line[1].split(' '))
                self.preprocessed_data[line[0]] = {
                    "string": self.docs_s_tokenized[-1],
                    "list": self.docs_l_tokenized[-1]
                }

        new_order = [None] * len(self.preprocessed_data)

        indices = list(self.preprocessed_data.keys())

        # Reorder the titles in the same order as preprocessed_data is saved
        for title in self.titles:
            pos = indices.index(title.lower())
            new_order[pos] = title

        self.titles = new_order
        self.preprocess_done = True
