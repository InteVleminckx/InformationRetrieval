#include "preprocessor.cpp"

extern "C" {

    void dpp(const char* input_path, const char* output_path, const int num_processes) {
        preprocess_data(input_path, output_path, num_processes);
    }

}