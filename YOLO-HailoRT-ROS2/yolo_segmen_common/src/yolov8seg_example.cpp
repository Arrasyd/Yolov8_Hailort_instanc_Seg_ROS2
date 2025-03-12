#include "hailo/hailort.hpp"
#include "yolo_segmen_common/common.h"

#include "yolo_segmen_common/hailo_objects.hpp"
#include "yolo_segmen_common/yolov8seg_postprocess.hpp"
#include "yolo_segmen_common/detection_inference.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <random>

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
std::mutex m;


using namespace hailort;

namespace yolo_cpp {

    void print_inference_statistics(std::chrono::duration<double> inference_time,
                                    std::string hef_file, double frame_count) { 
        std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
        std::cout << "-I- " << hef_file.substr(0, hef_file.find(".")) << std::endl;
        std::cout << "-I-----------------------------------------------" << std::endl;
        std::cout << "\n-I-----------------------------------------------" << std::endl;
        std::cout << "-I- Inference & Postprocess                        " << std::endl;
        std::cout << "-I-----------------------------------------------" << std::endl;
        std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
        std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
        std::cout << "-I- Latency:      " << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
        std::cout << "-I-----------------------------------------------" << std::endl;
    }


    std::string info_to_str(hailo_vstream_info_t vstream_info) {
        std::string result = vstream_info.name;
        result += " (";
        result += std::to_string(vstream_info.shape.height);
        result += ", ";
        result += std::to_string(vstream_info.shape.width);
        result += ", ";
        result += std::to_string(vstream_info.shape.features);
        result += ")";
        return result;
    }


    //template <typename T>
    std::vector<Object> YoloHailoRT::post_processing_all(std::vector<std::shared_ptr<FeatureData<u_int8_t>>> &features, size_t frame_count, 
                                    std::vector<cv::Mat>& frames, double org_height, double org_width, bool nms_on_hailo, std::string model_type) {

        auto status = HAILO_SUCCESS;
        std::vector<Object> objects;    
        std::sort(features.begin(), features.end(), &FeatureData<u_int8_t>::sort_tensors_by_size);
        std::cout<<"selesai read all 1"<<std::endl;
        std::random_device rd;
        std::cout<<"selesai read all 2"<<std::endl;
        std::mt19937 gen(rd());
        std::cout<<"selesai read all 3"<<std::endl;
        std::uniform_int_distribution<> random_index(0, COLORS.size() - 1);

        // cv::VideoWriter video("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size((int)org_width, (int)org_height));
        std::cout<<"selesai read all 4"<<std::endl;
        m.lock();
        std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;
        m.unlock();

        for (size_t i = 0; i < frame_count; i++){
            HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
            std::cout<<"selesai read all 5"<<std::endl;
            for (uint j = 0; j < features.size(); j++) {
                roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<u_int8_t*>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
                std::cout<<"selesai read all 6"<<std::endl;
            }

            auto filtered_masks = filter(roi, (int)org_height, (int)org_width);
            std::cout<<"selesai read all 7"<<std::endl;
        
            for (auto &feature : features) {
                feature->m_buffers.release_read_buffer();
                std::cout<<"selesai read all 8"<<std::endl;
            }
            std::cout<<"selesai read all 9"<<std::endl;
            std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
            std::cout<<"selesai read all 10"<<std::endl;
            cv::resize(frames[0], frames[0], cv::Size((int)org_width, (int)org_height), 1);
            std::cout<<"selesai read all 11"<<std::endl;

            for (auto &detection : detections) {
                if (detection->get_confidence()==0) {
                    continue;
                    std::cout<<"selesai read all 12"<<std::endl;
                }
                std::cout<<"selesai read all 13"<<std::endl;
                HailoBBox bbox = detection->get_bbox();
                std::cout<<"selesai read all 14"<<std::endl;
                Object obj;
                std::cout<<"selesai read all 15"<<std::endl;
                obj.rect = cv::Rect_<float>(bbox.xmin() * float(org_width), bbox.ymin() * float(org_height), 
                                        (bbox.xmax() - bbox.xmin()) * float(org_width), (bbox.ymax() - bbox.ymin()) * float(org_height));
                std::cout<<"selesai read all 16"<<std::endl;
                obj.label = detection->get_class_id() - 1;
                std::cout<<"selesai read all 17"<<std::endl;
                obj.prob = detection->get_confidence();
                std::cout<<"selesai read all 18"<<std::endl;
                if (obj.prob > 0.6) {
                     objects.push_back(obj);
                     std::cout<<"selesai read all 19"<<std::endl;
                    }
                
                cv::rectangle(frames[0], cv::Point2f(bbox.xmin() * float(org_width), bbox.ymin() * float(org_height)), 
                            cv::Point2f(bbox.xmax() * float(org_width), bbox.ymax() * float(org_height)), 
                            cv::Scalar(0, 0, 255), 1);
                
                std::cout << "Detection: " << detection->get_label() << ", Confidence: " << std::fixed << std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;
            }

            for (auto& mask : filtered_masks){
                cv::Mat overlay = cv::Mat(frames[0].rows, frames[0].cols, CV_8UC3, cv::Scalar(0));
                auto pixel_color = COLORS[random_index(gen)];
                for (int r = 0; r < mask.rows; r++) {
                    for (int c = 0; c < mask.cols ; c++) {
                        if (mask.at<float>(r, c) > 0.7) {
                            overlay.at<cv::Vec3b>(r,c) = pixel_color;
                        }
                    }
                }
                cv::addWeighted(frames[0], 1, overlay, 0.5, 0.0, frames[0]);
                overlay.release();
                mask.release();
            }

            // cv::imshow("Display window", frames[0]);
            // cv::waitKey(0);

            // video.write(frames[0]);
            cv::imwrite("output_image.jpg", frames[0]);
            
            frames[0].release();
            
            m.lock();
            frames.erase(frames.begin());
            m.unlock();
        }


        return objects;
    }

    //template <typename T>
    hailo_status read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData<u_int8_t>> feature, size_t frame_count) { 

        m.lock();
        std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
        m.unlock(); 

        std::vector<u_int8_t>& buffer = feature->m_buffers.get_write_buffer();
        hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
        feature->m_buffers.release_write_buffer();
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }


        return HAILO_SUCCESS;
    }

    hailo_status use_single_frame(InputVStream& input_vstream, std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                                    std::vector<cv::Mat>& frames, cv::Mat& image, int frame_count){

        hailo_status status = HAILO_SUCCESS;
        write_time_vec = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < frame_count; i++) {
            m.lock();
            frames.push_back(image);
            m.unlock();
            status = input_vstream.write(MemoryView(frames[frames.size() - 1].data, input_vstream.get_frame_size()));
            if (HAILO_SUCCESS != status)
                return status;
        }

        return HAILO_SUCCESS;
    }


    hailo_status write_all(InputVStream& input_vstream, std::string input_path, 
                    std::vector<cv::Mat>& frames, std::string& cmd_num_frames, const cv::Mat &frame)  {
        m.lock();
        std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
        m.unlock();

        hailo_status status = HAILO_SUCCESS;
        
        auto input_shape = input_vstream.get_info().shape;
        int height = input_shape.height;
        int width = input_shape.width;
        


        for(;;) {
            //cv::resize(frame, frame, cv::Size(width, height), 1);
            m.lock();
            frames.push_back(frame);
            m.unlock();

            input_vstream.write(MemoryView(frames[frames.size() - 1].data, input_vstream.get_frame_size())); // Writing height * width, 3 channels of uint8
            if (HAILO_SUCCESS != status)
                return status;
            
            //frame.release();
        }
        return HAILO_SUCCESS;
    }
    

    template <typename T>
    hailo_status create_feature(hailo_vstream_info_t vstream_info, size_t output_frame_size, std::shared_ptr<FeatureData<T>> &feature) {
        feature = std::make_shared<FeatureData<T>>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
            vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

        return HAILO_SUCCESS;
    }

    //template <typename T>
    std::vector<Object> YoloHailoRT::run_inference(std::vector<InputVStream>& input_vstream, std::vector<OutputVStream>& output_vstreams, std::string input_path,
                        size_t frame_count, std::string cmd_img_num, const cv::Mat &frame) {

        hailo_status status = HAILO_UNINITIALIZED;

        double org_height = frame.rows;
        double org_width = frame.cols;

        std::string model_type = "";
        
        auto output_vstreams_size = output_vstreams.size();

        bool nms_on_hailo = false;
        std::string output_name = (std::string)output_vstreams[0].get_info().name;
        if (output_vstreams_size == 1 && (output_name.find("nms") != std::string::npos)) {
            nms_on_hailo = true;
            model_type = output_name.substr(0, output_name.find('/'));
        }

        std::vector<std::shared_ptr<FeatureData<u_int8_t>>> features;
        features.reserve(output_vstreams_size);
        for (size_t i = 0; i < output_vstreams_size; i++) {
            std::shared_ptr<FeatureData<u_int8_t>> feature(nullptr);
            auto status = create_feature(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);

            features.emplace_back(feature);
        }

        std::vector<cv::Mat> frames;

        // Create the write thread
        auto input_thread(std::async(write_all, std::ref(input_vstream[0]), input_path, std::ref(frames), std::ref(cmd_img_num), frame));

        // Create read threads
        std::vector<std::future<hailo_status>> output_threads;
        output_threads.reserve(output_vstreams_size);
        for (size_t i = 0; i < output_vstreams_size; i++) {
            output_threads.emplace_back(std::async(read_all, std::ref(output_vstreams[i]), features[i], frame_count)); 
        }

        // Create the postprocessing thread
        auto pp_thread(std::async(&YoloHailoRT::post_processing_all,this, std::ref(features), frame_count, std::ref(frames), org_height, org_width, nms_on_hailo, model_type));

        
        for (size_t i = 0; i < output_threads.size(); i++) {
            status = output_threads[i].get();
        }
        auto input_status = input_thread.get();
        auto object = pp_thread.get();

        return object; //post_processing_all(std::ref(features), frame_count, std::ref(frames), org_height, org_width, nms_on_hailo, model_type);
    }

    Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &vdevice, std::string yolov_hef)
    {
        auto hef_exp = Hef::create(yolov_hef);
        if (!hef_exp) {
            return make_unexpected(hef_exp.status());
        }
        auto hef = hef_exp.release();

        auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!configure_params) {
            return make_unexpected(configure_params.status());
        }

        auto network_groups = vdevice.configure(hef, configure_params.value());
        if (!network_groups) {
            return make_unexpected(network_groups.status());
        }

        if (1 != network_groups->size()) {
            std::cerr << "Invalid amount of network groups" << std::endl;
            return make_unexpected(HAILO_INTERNAL_FAILURE);
        }

        return std::move(network_groups->at(0));
    }

    std::string getCmdOption(int argc, char *argv[], const std::string &option)
    {
        std::string cmd;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (0 == arg.find(option, 0))
            {
                std::size_t found = arg.find("=", 0) + 1;
                cmd = arg.substr(found);
                return cmd;
            }
        }
        return cmd;
    }

    std::vector<Object> YoloHailoRT::entry(const cv::Mat &frames) {

        hailo_status status = HAILO_UNINITIALIZED;
        std::string yolov_hef      = "/home/bfc/Documents/Hailo-Application-Code-Examples/runtime/cpp/instance_segmentation/yolov8seg/yolov8s_seg.hef";
        std::string input_path     = "";
        std::string image_num      = "";

        auto vdevice_exp = VDevice::create();
        auto vdevice = vdevice_exp.release();
        auto network_group_exp = configure_network_group(*vdevice, yolov_hef);
        auto network_group = network_group_exp.release();
        auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
        auto vstreams = vstreams_exp.release();

        size_t frame_count;
        frame_count = -1;

        auto object = run_inference(std::ref(vstreams.first), 
                            std::ref(vstreams.second), 
                            input_path, frame_count, image_num, frames);      

        return object;
    }


}
