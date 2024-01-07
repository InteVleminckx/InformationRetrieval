//
// Created by inte on 17/12/23.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <future>
#include <regex>
#include <locale>
#include <unordered_set>
#include "stemmer/english_stem.h"
#include <codecvt>

using namespace std;

/**
 * @function: isStopword - checks if the token is a stopword or not
 * @param match: the token to be checked
 * @return: a boolean indicating if it is a stopword or not
 */
bool isStopword(const std::string &match) {
    static const std::unordered_set<std::string> stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've",
            "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
            "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
            "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
            "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re",
            "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
            "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma",
            "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
            "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn",
            "wouldn't"
    };

    return stopwords.find(match) != stopwords.end();
}

/**
 * @function: checkWord: checks if the token is a stop word or not and returns the stemmed form
 * @param match: the token to be checked
 * @return if no stop word, the stemmed version of the token. Otherwise an empty string
 */
string checkWord(const string &match) {

    // The match can just be an n, this is because with the regex all newline chars transform to a n
    // So we need to remove them
    if (match == "n") return "";

    // Check if it is a stop word, if this is the case, skip this word
    if (isStopword(match)) return "";

    // Creating an english stemmer
    stemming::english_stem<std::wstring> stemmer;
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

    // The stemmer expect a wstring type as input, so first convert the string
    wstring w_stringed = converter.from_bytes(match);
    stemmer(w_stringed);

    // Return the stemmed word and add a space
    string stemmed = converter.to_bytes(w_stringed);
    return stemmed + " ";
}

/**
 *  @function: processLines - Processes each line / document for a vector of documents
 * @param lines: the documents / lines to be processed
 * @param process_no: the number of the process
 * @return: the processed documents linked to their process id
 */
pair<int, vector<string>> processLines(const vector<string> &lines, const int process_no) {

    pair<int, vector<string>> finished_lines = {process_no, {}};

    // regex for extracting words
    regex remove_thrash("\\b[[:alpha:]]+\\b");

    // Looping over the lines
    for (const auto &line: lines) {
        string _line = line;

        // Getting the title
        size_t pos_comma = line.find(",\"[[");

        string title = line.substr(0, pos_comma);
        if (title[0] == '\"') {
            // Remove first character from the string
            title.erase(0, 1);
        }
        if (title[title.size() - 1] == '\"') {
            // Remove last character from the string
            title.erase(title.size() - 1);
        }
        std::sregex_iterator iterator(_line.begin() + pos_comma, _line.end(), remove_thrash);
        std::sregex_iterator end;

        string newline;

        // Iterate over the matches
        while (iterator != end) {
            // Apply some preprocessing stuff on the word / token
            string word = iterator->str();
            newline += checkWord(word);
            ++iterator;
        }

        // Saving the new preprocessed document
        string finished_line = title + "," + newline + "\n";
        finished_lines.second.push_back(finished_line);

    }

    return finished_lines;

}

/**
 * @function: preprocess_data - divides the documents over multiples processes and starts the preprocessing proces
 * @param input_file: the input file that contains the documents
 * @param output_file: the output file where the preprocessed documents need to be written in
 * @param num_processes: the number of processes that will be used for preprocessing
 */
void preprocess_data(const char *input_file, const char *output_file, const int num_processes) {

    // Creating variables
    string line;
    vector<vector<string>> linesByProcess(num_processes);
    vector<future<pair<int, vector<string>>>> futures;
    vector<pair<int, vector<string>>> finished_processes;

    // Reading out input file
    ifstream file(input_file);

    // Get the number of lines in the document
    int lineCount = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');

    // Calculate the number of lines per document
    int no_of_lines = lineCount / num_processes;
    ifstream input(input_file);

    int process_index = 0;
    while (getline(input, line)) {
        // Skip title section line
        if (line == "title,sections") continue;

        // At the line to the current selected process
        linesByProcess[process_index].push_back(line);

        // If the size of the vector contains the number of lines per process, then start filling up the next process vector
        // and fill up the last process with the remainder lines
        if (linesByProcess[process_index].size() == no_of_lines && process_index < num_processes - 1)
            process_index += 1;

    }

    ofstream csv(output_file);

    // Create threads for each process
    futures.reserve(num_processes);

    // Calling the process lines function for each process
    for (int process = 0; process < num_processes; ++process) {
        futures.emplace_back(async(launch::async, processLines, linesByProcess[process], process));
    }


    // Wait for all threads to finish
    finished_processes.reserve(futures.size());

    // Storing the outputs
    for (auto &future: futures) {
        finished_processes.push_back(future.get());
    }


    // Write the preprocessed documents to a csv file in the same order it was read out
    int process_count = 0;
    while (process_count < num_processes) {
        for (const auto &it: finished_processes) {
            if (it.first == process_count) {
                for (const auto &doc: it.second) {
                    csv << doc;
                }
                process_count += 1;
            }
        }
    }
}