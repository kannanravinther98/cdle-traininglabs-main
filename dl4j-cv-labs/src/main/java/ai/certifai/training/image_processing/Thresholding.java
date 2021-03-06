/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.certifai.training.image_processing;

import freemarker.ext.jsp.TaglibFactory;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

/*
 * TASKS:
 * -----
 * 1. Load and display sat_map3.jpg from the resources/image_processing folder
 * 2. Apply threshold(=50) to the image
 * 3. Display the thresholded image
 * * Change the threshold value to observe the effects
 *
 * */

public class Thresholding {

    public static void main(String[] args) throws IOException {
        String imgFilePath = new ClassPathResource("image_processing/sat_map3.jpg").getFile().getAbsolutePath();
        Mat myImg = imread(imgFilePath);

        Display.display(myImg, "Original image");

        Mat thresh = new Mat();


        threshold (myImg, thresh, 180, 255,THRESH_BINARY);
        Display.display(thresh, "With thresholding");
    }


}
