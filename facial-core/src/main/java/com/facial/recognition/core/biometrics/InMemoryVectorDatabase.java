package com.facial.recognition.core.biometrics;

import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.RecognitionResult;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Thread-safe in-memory vector database implementing fast nearest-neighbor search
 * over high-dimensional biometric facial embeddings.
 */
@Slf4j
public class InMemoryVectorDatabase implements VectorDatabase {

    private final Map<String, BiometricProfile> profiles = new ConcurrentHashMap<>();

    @Override
    public void enroll(BiometricProfile profile) {
        if (profile == null || profile.getId() == null) {
            throw new IllegalArgumentException("Cannot enroll null profile or profile with null ID");
        }
        profiles.put(profile.getId(), profile);
        log.info("Profile enrolled successfully: id='{}', name='{}', embeddings_count={}",
                profile.getId(), profile.getName(), profile.getEmbeddings().size());
    }

    @Override
    public boolean remove(String profileId) {
        if (profileId == null) return false;
        boolean removed = profiles.remove(profileId) != null;
        if (removed) {
            log.info("Profile id='{}' removed from vector database.", profileId);
        }
        return removed;
    }

    @Override
    public Optional<BiometricProfile> findById(String profileId) {
        if (profileId == null) return Optional.empty();
        return Optional.ofNullable(profiles.get(profileId));
    }

    @Override
    public List<BiometricProfile> getAllProfiles() {
        return new ArrayList<>(profiles.values());
    }

    @Override
    public int size() {
        return profiles.size();
    }

    @Override
    public void clear() {
        profiles.clear();
        log.info("Vector database cleared.");
    }

    @Override
    public RecognitionResult verify(String claimedProfileId, float[] probeEmbedding, double similarityThreshold) {
        long startTime = System.currentTimeMillis();
        if (claimedProfileId == null || probeEmbedding == null || probeEmbedding.length == 0) {
            return RecognitionResult.unknown(null, System.currentTimeMillis() - startTime);
        }

        BiometricProfile profile = profiles.get(claimedProfileId);
        if (profile == null || !profile.isActive() || profile.getEmbeddings().isEmpty()) {
            return RecognitionResult.unknown(null, System.currentTimeMillis() - startTime);
        }

        double maxSimilarity = -1.0;
        double minDistance = Double.MAX_VALUE;

        for (float[] enrolledEmbedding : profile.getEmbeddings()) {
            double sim = BiometricDistanceCalculator.cosineSimilarity(probeEmbedding, enrolledEmbedding);
            double dist = BiometricDistanceCalculator.euclideanDistance(probeEmbedding, enrolledEmbedding);

            if (sim > maxSimilarity) {
                maxSimilarity = sim;
                minDistance = dist;
            }
        }

        boolean matched = maxSimilarity >= similarityThreshold;
        long elapsed = System.currentTimeMillis() - startTime;

        return RecognitionResult.builder()
                .subjectId(profile.getId())
                .subjectName(profile.getName())
                .department(profile.getDepartment())
                .role(profile.getRole())
                .matched(matched)
                .similarity(maxSimilarity)
                .distance(minDistance)
                .detectionConfidence(1.0f)
                .inferenceTimeMs(elapsed)
                .build();
    }

    @Override
    public RecognitionResult identify(float[] probeEmbedding, double similarityThreshold) {
        List<RecognitionResult> topResults = searchTopK(probeEmbedding, 1, similarityThreshold);
        if (topResults.isEmpty() || !topResults.get(0).isMatched()) {
            long elapsed = topResults.isEmpty() ? 0 : topResults.get(0).getInferenceTimeMs();
            return RecognitionResult.unknown(null, elapsed);
        }
        return topResults.get(0);
    }

    @Override
    public List<RecognitionResult> searchTopK(float[] probeEmbedding, int k, double similarityThreshold) {
        long startTime = System.currentTimeMillis();
        if (probeEmbedding == null || probeEmbedding.length == 0 || profiles.isEmpty() || k <= 0) {
            return List.of();
        }

        // Min-heap to keep Top-K elements by similarity
        PriorityQueue<RecognitionResult> heap = new PriorityQueue<>(k,
                Comparator.comparingDouble(RecognitionResult::getSimilarity));

        for (BiometricProfile profile : profiles.values()) {
            if (!profile.isActive() || profile.getEmbeddings().isEmpty()) {
                continue;
            }

            double bestSimilarity = -1.0;
            double bestDistance = Double.MAX_VALUE;

            for (float[] enrolledEmb : profile.getEmbeddings()) {
                double sim = BiometricDistanceCalculator.cosineSimilarity(probeEmbedding, enrolledEmb);
                double dist = BiometricDistanceCalculator.euclideanDistance(probeEmbedding, enrolledEmb);
                if (sim > bestSimilarity) {
                    bestSimilarity = sim;
                    bestDistance = dist;
                }
            }

            boolean matched = bestSimilarity >= similarityThreshold;

            RecognitionResult res = RecognitionResult.builder()
                    .subjectId(profile.getId())
                    .subjectName(profile.getName())
                    .department(profile.getDepartment())
                    .role(profile.getRole())
                    .matched(matched)
                    .similarity(bestSimilarity)
                    .distance(bestDistance)
                    .inferenceTimeMs(System.currentTimeMillis() - startTime)
                    .build();

            if (heap.size() < k) {
                heap.offer(res);
            } else if (res.getSimilarity() > heap.peek().getSimilarity()) {
                heap.poll();
                heap.offer(res);
            }
        }

        // Drain heap into list sorted descending
        List<RecognitionResult> results = new ArrayList<>(heap);
        results.sort((a, b) -> Double.compare(b.getSimilarity(), a.getSimilarity()));

        // Update elapsed time
        long totalElapsed = System.currentTimeMillis() - startTime;
        for (RecognitionResult r : results) {
            r.setInferenceTimeMs(totalElapsed);
        }

        return results;
    }
}
