/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "yolo"

__BEGIN_DECLS
std::vector<cv::Mat> filter(HailoROIPtr roi, int org_image_height, int org_image_width)
__END_DECLS
