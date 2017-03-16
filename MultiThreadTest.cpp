#include <stdio.h>

#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

#include <chrono>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

const mx_float DEFAULT_MEAN = 117.0;

using namespace std::chrono;

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

void GetImageFile(const std::string image_file,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
               mean_data++;
            }
            if (channels > 1) {
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(std::string synset_file) {
    std::ifstream fi(synset_file.c_str());

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
    synset[best_idx].c_str(), best_idx, best_accuracy);
}

std::vector<mx_float> image_as_floats;
std::vector<PredictorHandle> predictors;

#define NUM_INFERENCES 10

void thread_inference_from_array(int index) {
	for(int i=0; i<NUM_INFERENCES; i++) {

		high_resolution_clock::time_point time_start = high_resolution_clock::now();

		// Set Input Image
		MXPredSetInput(predictors[index], "data", image_as_floats.data(), image_as_floats.size());

		// Do Predict Forward
		MXPredForward(predictors[index]);

		mx_uint output_index = 0;

		mx_uint *shape = 0;
		mx_uint shape_len;

		// Get Output Result
		MXPredGetOutputShape(predictors[index], output_index, &shape, &shape_len);

		size_t size = 1;
		for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

		std::vector<float> data(size);

		MXPredGetOutput(predictors[index], output_index, &(data[0]), size);

		high_resolution_clock::time_point time_end = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);

		std::cout << "Thread " << index << ", inference: " << i << " took " << time_span.count() << " sec" << std::endl;
	}
}

void initialize(int num_threads, int dev_type, const std::string& json_file,
		const std::string& param_file, const std::string& image_file) {

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };
    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        assert(false);
    }

	predictors.resize(num_threads);

    // Create Predictors
	for(int i=0; i<num_threads; i++) {
	    MXPredCreate((const char*)json_data.GetBuffer(),
	                 (const char*)param_data.GetBuffer(),
	                 static_cast<size_t>(param_data.GetLength()),
	                 dev_type,
	                 dev_id,
	                 num_input_nodes,
	                 input_keys,
	                 input_shape_indptr,
	                 input_shape_data,
	                 &predictors[i]);
	    assert(predictors[i]);
	}

    int image_size = width * height * channels;

    // Read Mean Data
    const mx_float* nd_data = NULL;

    image_as_floats.resize(image_size);

    GetImageFile(image_file, image_as_floats.data(),
                 channels, cv::Size(width, height), nd_data);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict dev_type num_threads apple.jpg json_file params_file" << std::endl;
        return 0;
    }

    int dev_type = strtol(argv[1], nullptr, 10);

    int num_threads = strtol(argv[2], nullptr, 10);

    std::string image_file;
    image_file = std::string(argv[3]);

    // Models path for your model, you have to modify it
    std::string json_file = std::string(argv[4]);
    std::string param_file = std::string(argv[5]);

    initialize(num_threads, dev_type, json_file, param_file, image_file);

    boost::thread threads[num_threads];
    for(int i=0; i<num_threads; i++) {
    	threads[i] = boost::thread(thread_inference_from_array, i);
    }
    for(int i=0; i<num_threads; i++) {
    	threads[i].join();
    }

    std::cout << "Done" << std::endl;

    for(int i=0; i<num_threads; i++) {
    	MXPredFree(predictors[i]);
    }
}

int main_old(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
        return 0;
    }

    std::string test_file;
    test_file = std::string(argv[1]);

    // Models path for your model, you have to modify it
    std::string json_file = "/home/indu/reko-data/Inception-BN-symbol.json";
    std::string param_file = "/home/indu/reko-data/Inception-BN-0126.params";
    std::string synset_file = "/home/indu/reko-data/synset.txt";
    std::string nd_file = "model/Inception/mean_224.nd";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };
    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

    // Create Predictor
    MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    int image_size = width * height * channels;

    // Read Mean Data
    const mx_float* nd_data = NULL;
    NDListHandle nd_hnd = 0;
    BufferFile nd_buf(nd_file);

    if (nd_buf.GetLength() > 0) {
        mx_uint nd_index = 0;
        mx_uint nd_len;
        const mx_uint* nd_shape = 0;
        const char* nd_key = 0;
        mx_uint nd_ndim = 0;

        MXNDListCreate((const char*)nd_buf.GetBuffer(),
                   nd_buf.GetLength(),
                   &nd_hnd, &nd_len);

        MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
    }

    // Read Image Data
    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

    GetImageFile(test_file, image_data.data(),
                 channels, cv::Size(width, height), nd_data);

    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), image_size);

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);

    // Release NDList
    if (nd_hnd)
      MXNDListFree(nd_hnd);

    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset(synset_file);

    // Print Output Data
    PrintOutputResult(data, synset);

    return 0;
}
