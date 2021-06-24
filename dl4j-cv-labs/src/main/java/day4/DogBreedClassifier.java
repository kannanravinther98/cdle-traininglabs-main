package day4;

        import ai.certifai.training.classification.DogBreedDataSetIterator;
        import org.datavec.image.transform.CropImageTransform;
        import org.datavec.image.transform.FlipImageTransform;
        import org.datavec.image.transform.PipelineImageTransform;
        import org.datavec.image.transform.RotateImageTransform;
        import org.deeplearning4j.nn.conf.layers.DenseLayer;
        import org.deeplearning4j.nn.conf.layers.OutputLayer;
        import org.deeplearning4j.nn.graph.ComputationGraph;
        import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
        import org.deeplearning4j.nn.transferlearning.TransferLearning;
        import org.deeplearning4j.nn.weights.WeightInit;
        import org.deeplearning4j.zoo.model.VGG16;
        import org.nd4j.linalg.activations.Activation;
        import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
        import org.nd4j.linalg.learning.config.Sgd;
        import org.nd4j.linalg.lossfunctions.LossFunctions;

        import java.io.IOException;
        import java.util.Random;

public class DogBreedClassifier {

    private static final int epoch = 1;
    private static final int numClasses = 5;
    private static final int seed = 123;
    private static final Random rng = new Random(seed);


    public static void main(String[] args) throws IOException {

        /* 1. Image Augmentation
           2. Setup iterator
           3. Call the iterator out
           4. Transfer learning
               a. Pretrained model
               b. FineTuneConfig
               c. ComputationalGraph
           5. Train our model
         */

        PipelineImageTransform pipeline = new PipelineImageTransform.Builder()
                .addImageTransform(new CropImageTransform(rng, 50), 0.5)
                .addImageTransform(new RotateImageTransform(rng, 90), 0.5)
                .addImageTransform(new FlipImageTransform(5), 0.6)
                .build();

        pipeline.setShuffle(false);

        DogBreedDataSetIterator.setup(100, 80, pipeline);

        DataSetIterator trainIter = DogBreedDataSetIterator.trainIterator();
        DataSetIterator testIter = DogBreedDataSetIterator.testIterator();

        VGG16 vgg16 = VGG16.builder().build();
        ComputationGraph vgg16CG = (ComputationGraph) vgg16.initPretrained();
        System.out.println(vgg16CG.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .updater(new Sgd(0.001))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16CG)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexAndConnections("predictions")
                .nOutReplace("fc1", 500, WeightInit.XAVIER)
                .addLayer("fc3", new DenseLayer.Builder()
                        .nIn(500).nOut(50)
                        .activation(Activation.RELU)
                        .build(), "fc2")
                .addLayer("fc4", new DenseLayer.Builder()
                        .nIn(50).nOut(100)
                        .activation(Activation.RELU)
                        .build(), "fc3")
                .addLayer("newpredictions", new OutputLayer.Builder()
                        .nIn(100).nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build(), "fc4")
                .setOutputs("newpredictions")
                .setFeatureExtractor("fc3")
                .build();

        System.out.println(vgg16Transfer.summary());

        vgg16Transfer.init();
        vgg16Transfer.fit(trainIter, epoch);



    }

}