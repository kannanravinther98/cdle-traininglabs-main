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

package ai.certifai.solution.regression.medicalcostprediction;

import ai.certifai.solution.regression.PlotUtil;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MedicalCostPrediction {

    private static Logger log = LoggerFactory.getLogger(MedicalCostPrediction.class);

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         *  We would be using the Boston Housing Dataset to be our regression example.
         *  This dataset is obtained from https://www.kaggle.com/mirichoi0218/insurance
         *  The description of the attributes:
         *  age: age of primary beneficiary
         *  sex: insurance contractor gender, female, male
         *  bmi: Body mass index
         *  children: Number of children covered by health insurance / Number of dependents
         *  smoker: Smoking
         *  region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
         *  charges: Individual medical costs billed by health insurance
         *
         * */

        int batchSize = 128;
        double lr = 0.0015;
        int input = 9;
        int hidden = 1000;
        int output = 1;
        int nEpoch = 30;
        double reg = 0.0001;
        int seed = 428;

        /*
         *  STEP 1: DATA PREPARATION
         *
         * */

        //  Preparing the data
        File file = new ClassPathResource("medicalCost/insurance.csv").getFile();
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(file));

        // Declaring the feature names in schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("age")
                .addColumnCategorical("sex", Arrays.asList("female", "male"))
                .addColumnDouble("bmi")
                .addColumnInteger("children")
                .addColumnCategorical("smoker", Arrays.asList("yes", "no"))
                .addColumnCategorical("region", Arrays.asList("northeast", "southeast", "southwest", "northwest"))
                .addColumnDouble("charge")
                .build();
        System.out.println("Initial Schema: " + schema);

        // Building transform process schema
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToInteger("sex", "smoker")
                .categoricalToOneHot("region")
                .build();

        System.out.println("Final Schema: " + transformProcess.getFinalSchema());

        //  adding the original data to a list for later transform purpose
        List<List<Writable>> originalData = new ArrayList<>();
        while (recordReader.hasNext()) {
            List<Writable> data = recordReader.next();
            originalData.add(data);
        }

        // transform data into final schema
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, transformProcess);

        //  Preparing to split the dataset into training set and test set
        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader, transformedData.size(), 9, 9, true);

        DataSet dataSet = iterator.next();
        dataSet.shuffle();

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.7);

        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        INDArray features = train.getFeatures();
        System.out.println("\nFeature shape: " + features.shapeInfoToString() + "\n");

        //  Assigning dataset iterator for training purpose
        ViewIterator trainIter = new ViewIterator(train, batchSize);
        ViewIterator testIter = new ViewIterator(test, batchSize);


        /*
         *  STEP 2: MODEL TRAINING
         *
         * */

        //  Configuring the structure of the model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(reg)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(input)
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(output)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Initialize UI server for visualization model performance
        log.info("****************************************** UI SERVER **********************************************");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
        log.info("\n*************************************** TRAINING **********************************************\n");

        long timeX = System.currentTimeMillis();
        for (int i = 0; i < nEpoch; i++) {
            long time = System.currentTimeMillis();
            trainIter.reset();
            log.info("Epoch " + i);
            model.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");

            log.info("********************************** VALIDATING *************************************************");
            RegressionEvaluation evaluation = model.evaluateRegression(testIter);
            System.out.println(evaluation.stats());
        }
        long timeY = System.currentTimeMillis();
        log.info("\n******************** TOTAL TIME TAKEN: " + (timeY - timeX) + "ms ******************************\n");

        // Print out target values and predicted values
        log.info("\n*************************************** PREDICTION **********************************************");

        testIter.reset();

        INDArray targetLabels = test.getLabels();
        System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());

        INDArray predictions = model.output(testIter);
        System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");

        System.out.println("Target \t\t\t Predicted");


        for (int i = 0; i < targetLabels.rows(); i++) {
            System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
        }

        // Plot the target values and predicted values
        PlotUtil.visualizeRegression(targetLabels, predictions);

        // Print out model summary
        log.info("\n************************************* MODEL SUMMARY *******************************************");
        System.out.println(model.summary());

    }
}

