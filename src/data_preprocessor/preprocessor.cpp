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


bool isStopword(const std::string& match) {
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

string checkWord(const string& match) {

    if (match == "n") return "";
    if (isStopword(match)) return "";

    stemming::english_stem<std::wstring> stemmer;
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

    wstring w_stringed = converter.from_bytes(match);
    stemmer(w_stringed);

    string stemmed = converter.to_bytes(w_stringed);

    return match + " ";
}

void processLines(const vector<string> &lines, mutex &mtx, ofstream& csv) {

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
        } if (title[title.size() - 1] == '\"') {
            // Remove last character from the string
            title.erase(title.size() - 1);
        }
        std::sregex_iterator iterator(_line.begin() + pos_comma, _line.end(), remove_thrash);
        std::sregex_iterator end;

        string newline;

        // Iterate over the matches and print each word
        while (iterator != end) {

            string word = iterator->str();
            newline += checkWord(word);
            ++iterator;
        }

        lock_guard<mutex> lock(mtx);
        csv << title << "," << newline << "\n";

    }


}

void preprocess_data(const char *input_file, const char *output_file, const int num_processes) {

    // Creating variables
    string line;
    vector<vector<string>> linesByProcess(num_processes);
    vector<future<void>> futures;

    // Reading out input file
    ifstream input(input_file);

    // Divide lines over processes
    int process_index = 0;
    while (getline(input, line)) {
        linesByProcess[process_index].push_back(line);
        process_index = (process_index + 1) % num_processes;
    }

    // Remove Title, Sectîon from the first line
    linesByProcess[0].erase(linesByProcess[0].begin());
    mutex mtx;

    ofstream csv(output_file);

    // Create threads for each process
    for (int process = 0; process < num_processes; ++process) {
        futures.emplace_back(async(launch::async, processLines, linesByProcess[process], ref(mtx), ref(csv)));
    }

    // Wait for all threads to finish
    for (auto &future: futures) {
        future.wait();
    }

}

int main() {
    preprocess_data("../../data/lowered_input.csv", "../../data/pps.csv", 10);
}