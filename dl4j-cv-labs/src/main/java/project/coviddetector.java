package project;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class coviddetector {

    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static final String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    private static double TrainPerc = 0.7;
    private static int height = 80;
    private static int width = 80;
    private static int nChannels = 3;
    private static int batchSize = 30;
    private static int numClass = 2;
    private static double lr = 1e-3;   //1e-3, 1e-6, 1e-1
    private static int nEpoch = 10;



    public static void main(String[] args) throws IOException {
        File inputFile = new ClassPathResource("covidnormal").getFile();
        FileSplit filesplit = new FileSplit(inputFile, allowedFormats, rng);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter bPF = new BalancedPathFilter(rng, allowedFormats, labelMaker);

        InputSplit[] allData = filesplit.sample(bPF, TrainPerc, 1 - TrainPerc);
        InputSplit trainData = allData[0];
        InputSplit testData = allData[1];

        ImageRecordReader trainRR = new ImageRecordReader(height, width, nChannels, labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height, width, nChannels, labelMaker);

        trainRR.initialize(trainData);
        testRR.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numClass);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numClass);

        DataNormalization scalar = new ImagePreProcessingScaler();
        trainIter.setPreProcessor(scalar);
        testIter.setPreProcessor(scalar);

        MultiLayerConfiguration neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)// WeightInitialization
                .updater(new Adam(lr))// Updater
                // Regularization
                // Dropout


                .list()
                //.layer(1, new ConvolutionLayer.Builder()
                //input
                //Output
                //KernelSize
                //Stride
                //Activation Function
                //        .build())

                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(nChannels)//input
                        .nOut(32)//Output
                        .kernelSize(3,3)//KernelSize
                        .stride(1,1)//Stride
                        .activation(Activation.RELU)//Activation Function
                        .build())
                // output = floor (i + 2p -k)/s +1)
                //i = 3
                //k = 3
                //s = 1
                //p = 0
                .layer(1, new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        // .nIn(nChannels)//input
                        .nOut(64)//Output
                        .kernelSize(3,3)//KernelSize
                        .stride(1,1)//Stride
                        .activation(Activation.RELU)//Activation Function
                        .build())
                .layer(3, new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(4, new ConvolutionLayer.Builder()
                        // .nIn(nChannels)//input
                        .nOut(128)//Output
                        .kernelSize(3,3)//KernelSize
                        .stride(1,1)//Stride
                        .activation(Activation.RELU)//Activation Function
                        .build())
                .layer(5, new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
//                .layer(6, new ConvolutionLayer.Builder()
//                        // .nIn(nChannels)//input
//                        .nOut(256)//Output
//                        .kernelSize(3,3)//KernelSize
//                        .stride(1,1)//Stride
//                        .activation(Activation.RELU)//Activation Function
//                        .build())
//                .layer(7, new SubsamplingLayer.Builder()
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(8, new ConvolutionLayer.Builder()
//                        // .nIn(nChannels)//input
//                        .nOut(512)//Output
//                        .kernelSize(3,3)//KernelSize
//                        .stride(1,1)//Stride
//                        .activation(Activation.RELU)//Activation Function
//                        .build())
//                .layer(9, new SubsamplingLayer.Builder()
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(10, new ConvolutionLayer.Builder()
//                        // .nIn(nChannels)//input
//                        .nOut(1024)//Output
//                        .kernelSize(3,3)//KernelSize
//                        .stride(1,1)//Stride
//                        .activation(Activation.RELU)//Activation Function
//                        .build())
//                .layer(11, new SubsamplingLayer.Builder()
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
                .layer(6, new DenseLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(8, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(numClass)
                        .build())
                .setInputType(InputType.convolutional(height,width,nChannels))
                .build();

        MultiLayerNetwork cdmodel = new MultiLayerNetwork(neuralNetConfiguration);
       cdmodel.init();

        System.out.println(cdmodel.summary());



        Evaluation evalTrain = cdmodel.evaluate(trainIter);
        Evaluation evalTest = cdmodel.evaluate(testIter);

        System.out.println("Training Evaluation: "+ evalTrain.stats());
        System.out.println("Testing Evaluation: "+ evalTest.stats());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        cdmodel.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(10),
                new EvaluativeListener(trainIter,1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter,1, InvocationType.EPOCH_END)
        );

        //   rpsmodel.setListeners(new ScoreIterationListener(10));
        cdmodel.fit(trainIter,nEpoch);

    }
}
