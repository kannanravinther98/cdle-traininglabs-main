package day5;

        import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
        import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
        import org.datavec.api.split.FileSplit;
        import org.datavec.api.transform.TransformProcess;
        import org.datavec.api.transform.filter.FilterInvalidValues;
        import org.datavec.api.transform.schema.Schema;
        import org.datavec.api.writable.Writable;
        import org.datavec.local.transforms.LocalTransformExecutor;
        import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
        import org.deeplearning4j.nn.api.Model;
        import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
        import org.deeplearning4j.nn.graph.ComputationGraph;
        import org.deeplearning4j.zoo.ZooModel;
        import org.deeplearning4j.zoo.model.AlexNet;
        import org.deeplearning4j.zoo.model.LeNet;
        import org.nd4j.common.io.ClassPathResource;
        import org.nd4j.linalg.dataset.DataSet;
        import org.nd4j.linalg.dataset.SplitTestAndTrain;
        import org.nd4j.linalg.dataset.ViewIterator;
        import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

        import java.io.File;
        import java.io.IOException;
        import java.sql.SQLOutput;
        import java.util.ArrayList;
        import java.util.List;

public class DigitRecognizer {

    public static void main(String[] args) throws IOException, InterruptedException {

        File filePath = new ClassPathResource("mnistdigit/train.csv").getFile();
        CSVRecordReader csvRecordReader = new CSVRecordReader(1, ',');
        csvRecordReader.initialize(new FileSplit(filePath));

        Schema schema = getSchema();

        TransformProcess transformProcess = getTransformProcess(schema);
//        System.out.println(transformProcess.getFinalSchema());

        List<List<Writable>> data = new ArrayList<>();

        while(csvRecordReader.hasNext()) {
            data.add(csvRecordReader.next());
        }

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(data, transformProcess);

        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);
        RecordReaderDataSetIterator dataSetIter = new RecordReaderDataSetIterator(collectionRecordReader,
                transformedData.size(), 0, 10);

        DataSet dataSet = dataSetIter.next();
        dataSet.shuffle(123);
        SplitTestAndTrain split = dataSet.splitTestAndTrain(0.8);
        DataSet train = split.getTrain();
        DataSet test = split.getTest();

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(train);
        scaler.transform(train);
        scaler.transform(test);

        ViewIterator trainIter = new ViewIterator(train, 100);
        ViewIterator testIter = new ViewIterator(test, 100);

        ZooModel alexNet = AlexNet.builder().build();
        ComputationGraph alexNetModel = ((ComputationGraph) alexNet.init());
        System.out.println(alexNetModel.summary());



    }

    private static Schema getSchema() {

        return new Schema.Builder()
                .addColumnsInteger("label")
                .addColumnsInteger("pixel%d", 0, 783)
                .build();

    }

    private static TransformProcess getTransformProcess(Schema schema) {

        return new TransformProcess.Builder(schema)
                .filter(new FilterInvalidValues())
                .build();

    }

}