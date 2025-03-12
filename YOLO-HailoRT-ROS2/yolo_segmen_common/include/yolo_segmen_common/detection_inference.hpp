#ifndef HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_
#define HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#include "hailo/hailort.hpp"
#include "yolo_segmen_common/hailo_objects.hpp"
#include "yolo_segmen_common/yolov8seg_postprocess.hpp"
#include "yolo_segmen_common/double_buffer.hpp"
#include "yolo_segmen_common/object.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <thread>

//constexpr bool QUANTIZED = true;
//constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
using namespace hailort;

namespace yolo_cpp {

template <typename T>
class FeatureData {
public:
    FeatureData(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t width, hailo_vstream_info_t vstream_info) :
    m_buffers(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_width(width), m_vstream_info(vstream_info)
    {}
    static bool sort_tensors_by_size (std::shared_ptr<FeatureData> i, std::shared_ptr<FeatureData> j) { return i->m_width < j->m_width; };

    DoubleBuffer<T> m_buffers;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
    uint32_t m_width;
    hailo_vstream_info_t m_vstream_info;
};


class YoloHailoRT {
//typedef std::pair<std::vector<InputVStream>, std::vector<OutputVStream>> VStreams;
public:
    //explicit YoloHailoRT(const std::string &, const float, const float);
    //~YoloHailoRT() = default;

    //hailo_status init_device(const std::string &, std::shared_ptr<VDevice> &, std::shared_ptr<ConfiguredNetworkGroup> &, std::shared_ptr<VStreams> &);
    //template <typename T>
    std::vector<Object> post_processing_all(std::vector<std::shared_ptr<FeatureData<u_int8_t>>> &, size_t ,std::vector<cv::Mat>& , 
        double, double, bool, std::string );

    std::vector<Object> run_inference(std::vector<InputVStream>& input_vstream, std::vector<OutputVStream>& output_vstreams, std::string input_path,
        size_t frame_count, std::string cmd_img_num, const cv::Mat &frame);
    //std::vector<Object> post_processing(std::shared_ptr<FeatureData> &, cv::Mat &, const double, const double);

    
    //hailo_status read_all(OutputVStream &, std::shared_ptr<FeatureData>);
    //hailo_status create_feature(hailo_vstream_info_t, const size_t, std::shared_ptr<FeatureData> &);
    std::vector<Object> entry(const cv::Mat &);

    //Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &, const std::string &);

private:
    std::string m_model_path_;
    float m_nms_;
    float m_conf_;

    //std::shared_ptr<VDevice> m_device_;
    //std::shared_ptr<ConfiguredNetworkGroup> m_network_group_;
    //std::shared_ptr<VStreams> m_vstreams_;
};

} // namespace yolo_cpp

#endif // HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_
