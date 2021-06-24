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

package ai.certifai.training.datavec.loadcsv;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.writable.Writable;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.nio.charset.Charset;
import java.util.List;

public class LoadCSVDemo {
    private static File inputFile;
    public static void main(String[] args) throws Exception {
        inputFile = new ClassPathResource("datavec/titanic/train.csv").getFile();

        /*
        Exercise 1: Prepare titanic dataset
        - read csv file
        - define schema
        - define transform process
        - apply transform process
        - split into training and test dataset
        - normalize data
         */

        // read csv file
        /*
        YOUR CODE HERE
         */

        // define schema
        /*
        YOUR CODE HERE
         */

        // cabin has 687 missing values
        // embarked has 2 missing values
        // age has 177 missing values
        // define transform process
        /*
        - remove unused column (PassengerId, Name, Tiket, Cabin)
        - replace Age missing value with 0. hint: use .transform(new ReplaceInvalidWithIntegerTransform(...))
        - replace Embarked missing value with "S". hint: use .conditionalReplaceValueTransform(...))
        - convert Sex(category) to integer
        - convert Pclass, Embarked to one hot encoding
        */
        /*
        YOUR CODE HERE
        */

        //Process the data:
//        List<List<Writable>> originalData = new ArrayList<>();
//        while(rr.hasNext()){
//            originalData.add(rr.next());
//        }

        //Apply transform process
        /*
        YOUR CODE HERE
        */

        //Create iterator from processedData
//        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
//        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader,processedData.size(),0,2);
//        DataSet allData = iterator.next();

        //Shuffle and split data into training and test dataset
        /*
        YOUR CODE HERE
        */

        //Create iterator for splitted training and test dataset
//        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
//        DataSetIterator testIterator = new ViewIterator(testData, 2);

        //Normalize data to 0 - 1
        /*
        YOUR CODE HERE
        */

//        System.out.println("Sample of training vector: \n"+ trainIterator.next());

//        writeAndPrint(processedData);
    }

    private static void writeAndPrint(List<List<Writable>> processedData) throws Exception {
        // write processed data into file
        RecordWriter rw = new CSVRecordWriter();
        File outputFile = new File("titanicTransform.csv");
        if(outputFile.exists()) outputFile.delete();
        outputFile.createNewFile();

        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);
        rw.writeBatch(processedData);
        rw.close();

        //Print before + after:
        System.out.println("\n\n---- Original Data File ----");
        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);
    }


}
