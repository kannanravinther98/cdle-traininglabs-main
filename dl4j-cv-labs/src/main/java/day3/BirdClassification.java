package day3;

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
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.text.View;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//import static ai.certifai.training.regression.demandregression.RidershipDemandRegression.seed;

public class BirdClassification {
    public static void main(String[] args) throws IOException, InterruptedException {

        //Classification Problem
        // final int numberofClass = 6;
        final int seed = 123;
        final double splitratio = 0.75;
        final int numInput =10;
        final int numClass = 6;
        final double learningRate = 1e-1;
        final int epoch = 100;


        //Prepare the data
        File dataFile = new ClassPathResource("birdclassify/bird.csv").getFile();
        FileSplit fileSplit = new FileSplit(dataFile);

        //CSV record reader
        RecordReader crr = new CSVRecordReader(1,",");
        crr.initialize(fileSplit);

        //id,huml,humw,ulnal,ulnaw,feml,femw,tibl,tibw,tarl,tarw,type
        //Build the schema
        Schema inpschema = new Schema.Builder()
                .addColumnInteger("id")
                .addColumnsDouble("huml", "humw", "ulnal", "ulnaw", "feml", "femw", "tibl", "tibw", "tarl", "tarw")
                .addColumnCategorical("type", Arrays.asList("SW", "W", "T", "R", "P", "SO"))
                .build();

        System.out.println(inpschema);

        //Transform the schema
        TransformProcess transprocess = new TransformProcess.Builder(inpschema)
                .removeColumns("id")
                .categoricalToInteger("type")
                .build();
        Schema outschema = transprocess.getFinalSchema();

        System.out.println(outschema);

        List<List<Writable>> allData = new ArrayList<>();

        while (crr.hasNext()) {
            allData.add(crr.next());
        }
        List<List<Writable>> processData = LocalTransformExecutor.execute(allData, transprocess);

        //Iterator

        CollectionRecordReader collectRR = new CollectionRecordReader(processData);

        //Batchsize, label index, number of labels
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectRR, processData.size(), -1, 6);

        DataSet fullDataSet = dataSetIterator.next();
        fullDataSet.shuffle(seed);

        //Splitting
        SplitTestAndTrain tnt = fullDataSet.splitTestAndTrain(splitratio);

        DataSet trainData = tnt.getTrain();
        DataSet testData = tnt.getTest();


        //Data Normalization
        DataNormalization normalizedata = new NormalizerMinMaxScaler();
        normalizedata.fit(trainData);
        normalizedata.transform(trainData);


        //Model Configuration
        MultiLayerConfiguration netconfig = getConfig(numInput, numClass, learningRate);
        MultiLayerNetwork birdmodel = new MultiLayerNetwork(netconfig);
        birdmodel.init();

        //Set Listeners
        StatsStorage storage = new InMemoryStatsStorage();
        birdmodel.setListeners(new StatsListener(storage, 10));

        Evaluation eval;
        for(int i = 0; i<epoch; i++)
        {
            birdmodel.fit(trainData);
            eval = birdmodel.evaluate(new ViewIterator(testData, processData.size()));
            System.out.println("Epoch: "+ i +"Accuracy: " + eval.accuracy());
        }
    }

    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate)
    {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(32)
                        .nOut(numOutputs)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return config;
    }


}