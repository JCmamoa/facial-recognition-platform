package com.facial.recognition.persistence.repository;

import com.facial.recognition.persistence.entity.UserProfileEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Spring Data JPA repository for persisting and querying enrolled biometric user profiles.
 */
@Repository
public interface UserProfileRepository extends JpaRepository<UserProfileEntity, String> {

    List<UserProfileEntity> findByActiveTrue();

    Optional<UserProfileEntity> findByDocumentId(String documentId);

    List<UserProfileEntity> findByNameContainingIgnoreCase(String name);

    List<UserProfileEntity> findByDepartmentIgnoreCase(String department);
}
