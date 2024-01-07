#include "preprocessor.cpp"

void dpp(const char* input_path, const char* output_path, const int num_processes) {
    preprocess_data(input_path, output_path, num_processes);
}

int main(int argc, char * argv[]) {

    // Checks if there are four arguments where from 3 additional ones
    if (argc == 4) {

        // The input file
        char* input_file = argv[1];
        // The output file
        char* output_file = argv[2];
        // The number of processes to use
        int num_processes = stoi(argv[3]);
        // Call the data preprocessing function with the provided parameters
        dpp(input_file, output_file, num_processes);

        return 0;
    }

    return 1;

}