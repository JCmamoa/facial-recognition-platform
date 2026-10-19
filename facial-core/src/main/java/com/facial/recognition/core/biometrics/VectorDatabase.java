package com.facial.recognition.core.biometrics;

import com.facial.recognition.core.model.BiometricProfile;
import com.facial.recognition.core.model.RecognitionResult;

import java.util.List;
import java.util.Optional;

/**
 * Interface for vector database operations, enabling sub-millisecond nearest neighbor search,
 * profile storage, 1:1 biometric verification, and 1:N biometric identification.
 */
public interface VectorDatabase {

    /**
     * Enrolls or updates a subject profile with its facial signature vector(s).
     */
    void enroll(BiometricProfile profile);

    /**
     * Removes a profile by ID.
     */
    boolean remove(String profileId);

    /**
     * Finds a profile by ID.
     */
    Optional<BiometricProfile> findById(String profileId);

    /**
     * Returns all registered profiles.
     */
    List<BiometricProfile> getAllProfiles();

    /**
     * Returns total count of enrolled subjects.
     */
    int size();

    /**
     * Clears all enrolled vectors.
     */
    void clear();

    /**
     * Performs 1:1 biometric verification: compares probe against a claimed subject ID.
     */
    RecognitionResult verify(String claimedProfileId, float[] probeEmbedding, double similarityThreshold);

    /**
     * Performs 1:N biometric identification: searches the vector database for the closest subject.
     */
    RecognitionResult identify(float[] probeEmbedding, double similarityThreshold);

    /**
     * Searches for the Top-K nearest neighbors sorted by cosine similarity.
     */
    List<RecognitionResult> searchTopK(float[] probeEmbedding, int k, double similarityThreshold);
}
