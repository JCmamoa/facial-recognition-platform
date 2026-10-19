package com.facial.recognition.dl4j;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * DeepLearning4J (DL4J) Deep Neural Network for biometric facial signature classification.
 * Trains and evaluates multi-layer perceptrons directly on high-dimensional 128-d facial vectors.
 */
@Slf4j
public class BiometricNeuralClassifier {

    private final int inputDimension;
    private final int numClasses;
    private final MultiLayerNetwork model;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ClassificationOutput {
        private int predictedClass;
        private double confidence;
        private float[] probabilities;
    }

    public BiometricNeuralClassifier(int numClasses) {
        this(128, numClasses);
    }

    public BiometricNeuralClassifier(int inputDimension, int numClasses) {
        this.inputDimension = inputDimension;
        this.numClasses = Math.max(1, numClasses);
        this.model = buildNetwork();
        this.model.init();
        log.info("BiometricNeuralClassifier initialized (inputDim={}, numClasses={})", inputDimension, numClasses);
    }

    private MultiLayerNetwork buildNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputDimension)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DropoutLayer.Builder(0.2).build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(128)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return new MultiLayerNetwork(conf);
    }

    /**
     * Trains the neural classifier using supervised pairs of (embedding, classLabel).
     */
    public synchronized void train(List<float[]> embeddings, List<Integer> labels, int epochs) {
        if (embeddings == null || labels == null || embeddings.isEmpty() || embeddings.size() != labels.size()) {
            throw new IllegalArgumentException("Invalid training dataset for neural classifier");
        }

        int count = embeddings.size();
        INDArray features = Nd4jVectorUtils.toNd4jMatrix(embeddings);

        // One-hot encode labels
        INDArray oneHotLabels = Nd4j.zeros(count, numClasses);
        for (int i = 0; i < count; i++) {
            int label = labels.get(i);
            if (label >= 0 && label < numClasses) {
                oneHotLabels.putScalar(new int[]{i, label}, 1.0);
            }
        }

        DataSet dataSet = new DataSet(features, oneHotLabels);

        log.info("Beginning training BiometricNeuralClassifier for {} epochs on {} samples...", epochs, count);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            model.fit(dataSet);
            if (epoch % Math.max(1, epochs / 5) == 0 || epoch == epochs) {
                log.info("Epoch {}/{}: current loss = {}", epoch, epochs, model.score());
            }
        }
        log.info("Training complete. Final network score: {}", model.score());
    }

    /**
     * Predicts the most likely identity class and confidence score for a given facial embedding.
     */
    public ClassificationOutput predict(float[] embedding) {
        if (embedding == null || embedding.length == 0) {
            return new ClassificationOutput(-1, 0.0, new float[0]);
        }

        INDArray input = Nd4jVectorUtils.toNd4jVector(embedding);
        INDArray output = model.output(input);

        float[] probs = output.data().asFloat();
        int bestClass = 0;
        float maxProb = probs[0];

        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                bestClass = i;
            }
        }

        return ClassificationOutput.builder()
                .predictedClass(bestClass)
                .confidence(maxProb)
                .probabilities(probs)
                .build();
    }

    private BiometricNeuralClassifier(MultiLayerNetwork prebuiltModel, int inputDimension, int numClasses) {
        this.inputDimension = inputDimension;
        this.numClasses = numClasses;
        this.model = prebuiltModel;
    }

    /**
     * Serializes trained model weights to disk.
     */
    public void save(File file) throws IOException {
        ModelSerializer.writeModel(model, file, true);
        log.info("Saved BiometricNeuralClassifier model to: {}", file.getAbsolutePath());
    }

    /**
     * Loads a pre-trained model from disk.
     */
    public static BiometricNeuralClassifier load(File file) throws IOException {
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(file);
        int inDim = (int) net.layerInputSize(0);
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer outLayer = 
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) net.getLayerWiseConfigurations().getConf(net.getnLayers() - 1).getLayer();
        int outDim = (int) outLayer.getNOut();
        return new BiometricNeuralClassifier(net, inDim, outDim);
    }
}
