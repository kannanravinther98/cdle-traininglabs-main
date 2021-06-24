//package day3;
//
//import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
//import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
//import org.datavec.api.split.FileSplit;
//import org.datavec.api.transform.TransformProcess;
//import org.datavec.api.transform.schema.Schema;
//import org.datavec.local.transforms.LocalTransformExecutor;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.nd4j.common.io.ClassPathResource;
//import org.nd4j.evaluation.regression.RegressionEvaluation;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.dataset.DataSet;
//import org.nd4j.linalg.dataset.SplitTestAndTrain;
//import org.nd4j.linalg.dataset.ViewIterator;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.learning.config.Adam;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.List;
//
//public class bhouseprediction {
//    public static void main(String[] args) throws IOException, InterruptedException {
//        final int batchSize = 100;
//        final int input = 13;
//        final int output = 1;
//        final int nhidden1 = 64;
//        final int nhidden2 = 32;
//        final int nEpochs = 50;
//        final int seed = 123;
//        final double lr = 1e-1;
//
//        //Preparing the dataset
//        File dataFile = new ClassPathResource("boston/bostonHousing.csv").getFile();
//        CSVRecordReader CSVreader = new CSVRecordReader();
//        CSVreader.initialize(new FileSplit(dataFile));
//        //Declaring the schema
//        Schema inputSchema = new Schema.Builder()
//                .addColumnsDouble("CRIM", "ZN", "INDUS")
//                .addColumnInteger("CHAS")
//                .addColumnsDouble("NOX", "RM", "AGE", "DIS")
//                .addColumnInteger("RAD")
//                .addColumnsDouble("TAX","PTRATIO", "B", "LSTAT", "PRICE")
//                .build();
//        //Transform process
//        TransformProcess tp = new TransformProcess.Builder(inputSchema)
//                .build();
//        //Dataset iterator
//        List<List<Writable>> originalData = new ArrayList<>();
//        while(CSVreader.hasNext()){
//            List<Writable> data = CSVreader.next();
//            originalData.add(data);
//        }
//        List<List<Writable>> transformData = LocalTransformExecutor.execute(originalData,tp);
//        for (int i=0; i<transformData.size();i++){
//            System.out.println(transformData.get(i));
//        }
//        CollectionRecordReader crr = new CollectionRecordReader(transformData);
//        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transformData.size(), 13, 13, true);
//
//        //Datasplitting into train and test
//        DataSet allData = dataIter.next();
//        allData.shuffle();
//        SplitTestAndTrain testTrainSplit = allData.splitTestAndTrain(0.8);
//        DataSet trainSet = testTrainSplit.getTrain();
//        DataSet testSet = testTrainSplit.getTest();
//        ViewIterator trainIter = new ViewIterator(trainSet,batchSize);
//        ViewIterator testIter = new ViewIterator(testSet,batchSize);
//
//        //Model Configuration
//        //14 columns ( 13 features / input )  (1 - output)
//        //nIn - 13
//        //nOut - 1
//        //Activation - hidden , output
//        //loss function
//        MultiLayerConfiguration bconf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Adam(lr))
//                .list()
//                .layer(0, new DenseLayer.Builder()
//                        .nIn(input)
//                        .nOut(nhidden1)
//                        .activation(Activation.TANH)
//                        .build())
//                .layer(1, new DenseLayer.Builder()
//                        .nIn(nhidden1)
//                        .nOut(nhidden2)
//                        .activation(Activation.TANH)
//                        .build())
//                .layer(2, new OutputLayer.Builder()
//                        .nIn(nhidden2)
//                        .nOut(output)
//                        .activation(Activation.IDENTITY)
//                        .lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
//                        .build())
//                .build();
//
//        //Model initialization
//        MultiLayerNetwork bhousemodel = new MultiLayerNetwork(bconf);
//        bhousemodel.init();
//        bhousemodel.summary();
//
//        //SetListeners
//        bhousemodel.setListeners(new ScoreIterationListener(1));
//
//        //Model fitting
//        for(int j=0; j<nEpochs;j++) {
//            if(j%10==0)
//                System.out.println("Epoch: "+j);
//        }
//        bhousemodel.fit(trainIter);
//
//        //Model evaluation
//        RegressionEvaluation regEvel = bhousemodel.evaluateRegression(testIter);
//        System.out.println(regEvel.stats());
//    }
//}