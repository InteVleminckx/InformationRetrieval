#include "preprocessor.cpp"

void dpp(const char* input_path, const char* output_path, const int num_processes) {
    preprocess_data(input_path, output_path, num_processes);
}

int main(int argc, char * argv[]) {

    if (argc == 4) {

        char* input_file = argv[1];
        char* output_file = argv[2];
        int num_processes = stoi(argv[3]);

        dpp(input_file, output_file, num_processes);

        return 0;
    }

    return 1;

}