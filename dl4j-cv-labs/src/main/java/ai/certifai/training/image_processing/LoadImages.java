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

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.nativeblas.Nd4jCpu;

import java.io.IOException;
import java.nio.file.attribute.UserPrincipalLookupService;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

//        String myImg = new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath();
//         Mat src = imread (myImg);
//         Display.display(src,"This is lena");


        String myImg = new ClassPathResource("image_processing/opencv.png").getFile().getAbsolutePath();
        Mat src = imread (myImg);

        System.out.println("Image height:" + src.arrayHeight());
        System.out.println("Image weight:" + src.arrayWidth());

        Display.display(src,"This is OpenCV");

        Mat downsized = new Mat();
        Mat upsized_linear = new Mat();
        Mat upsized_cubic = new Mat();
        Mat upsized_nearest = new Mat();

        resize(src, downsized, new Size(500, 500));
        resize(downsized, upsized_linear, new Size(1470, 1200), 0,0,INTER_LINEAR);
        resize(downsized,upsized_cubic, new Size(1470, 1200),0,0,INTER_CUBIC);
        resize(downsized,upsized_nearest, new Size(1470, 1200),0,0,INTER_NEAREST);


        Display.display(downsized,"Downsized OpenCV");
        Display.display(upsized_linear, "upsized OpenCV using Linear");
        Display.display(upsized_cubic, "upsized OpenCV using Cubic ");
        Display.display(upsized_nearest, "upsized OpenCV using Nearest");

    }
}
