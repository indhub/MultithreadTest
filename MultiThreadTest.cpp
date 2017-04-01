#include <stdio.h>

#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

#include <chrono>
#include <boost/thread.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <opencv2/opencv.hpp>

const mx_float DEFAULT_MEAN = 117.0;

using namespace std::chrono;

boost::interprocess::interprocess_semaphore job_available(0);
boost::interprocess::interprocess_semaphore job_done(0);

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

void GetImageFile(const cv::Mat& im_ori,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
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

void GetImageFile(const std::string image_file,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    GetImageFile(im_ori, image_data, channels, resize_size, mean_data);
}

std::vector<mx_float> image_as_floats;
std::vector<PredictorHandle> predictors;

#define NUM_INFERENCES_PER_THREAD 10

void thread_inference_from_array(int index) {

	while(true) {

		job_available.wait();

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

		//std::cout << "Thread " << index << ", inference: " << i << " took " << time_span.count() << " sec" << std::endl;
		job_done.post();
	}
}

cv::Mat image_as_cv;

void thread_inference_from_mat(int index) {
	for(int i=0; i<NUM_INFERENCES_PER_THREAD; i++) {

		int width = 224;
	    int height = 224;
	    int channels = 3;

	    std::vector<mx_float> image_as_floats;

	    int image_size = width * height * channels;
	    const mx_float* nd_data = NULL;
	    image_as_floats.resize(image_size);
	    GetImageFile(image_as_cv, image_as_floats.data(),
	                 channels, cv::Size(width, height), nd_data);


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

void initialize(int dev_type, int num_threads, unsigned int batch_size, unsigned int num_non_zero, const std::string& json_file,
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
    const mx_uint input_shape_data[4] = { batch_size,
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
    const mx_float* nd_data = NULL;
    image_as_floats.resize(image_size*batch_size);

    for(int i=0; i<batch_size; i++) {
    	if(i<num_non_zero) {
			GetImageFile(image_file, image_as_floats.data() + (image_size*i),
						 channels, cv::Size(width, height), nd_data);
    	} else {
    		size_t zeros = (batch_size-i) * image_size;
    		std::fill(image_as_floats.begin() + (image_size*i), image_as_floats.end(), 0);
    		break;
    	}
    }

    image_as_cv = cv::imread(image_file, cv::IMREAD_COLOR);
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict dev_type num_threads batch_size num_non_zero apple.jpg resize? json_file params_file" << std::endl;
        return 0;
    }

    int dev_type = strtol(argv[1], nullptr, 10);

    int num_threads = strtol(argv[2], nullptr, 10);

    unsigned int batch_size = strtol(argv[3], nullptr, 10);

    unsigned int num_non_zero = strtol(argv[4], nullptr, 10);

    std::string image_file;
    image_file = std::string(argv[5]);

    bool should_resize = (strtol(argv[6], nullptr, 10) == 1);
    if(should_resize) {
    	std::cout << "Will include time to resize image" << std::endl;
    } else {
    	std::cout << "Will not include time to resize image" << std::endl;
    }

    // Models path for your model, you have to modify it
    std::string json_file = std::string(argv[7]);
    std::string param_file = std::string(argv[8]);

    initialize(dev_type, num_threads, batch_size, num_non_zero, json_file, param_file, image_file);

    boost::thread threads[num_threads];
    for(int i=0; i<num_threads; i++) {
    	if(should_resize) {
    		threads[i] = boost::thread(thread_inference_from_mat, i);
    	} else {
    		threads[i] = boost::thread(thread_inference_from_array, i);
    	}
    }

    while(true) {
    	high_resolution_clock::time_point time_start = high_resolution_clock::now();
    	for(int i=0; i<num_threads; i++) {
    		job_available.post();
    	}
    	for(int i=0; i<num_threads; i++) {
    		job_done.wait();
    	}
		high_resolution_clock::time_point time_end = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);

		std::cout << "All threads took " << time_span.count() << " sec" << std::endl;
    }

    std::cout << "Done" << std::endl;

    for(int i=0; i<num_threads; i++) {
    	MXPredFree(predictors[i]);
    }
}

