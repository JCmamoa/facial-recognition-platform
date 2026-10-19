package com.facial.recognition.api.controller;

import com.facial.recognition.api.FacialRecognitionApiApplication;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(classes = FacialRecognitionApiApplication.class)
@AutoConfigureMockMvc
class FaceRecognitionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    @DisplayName("GET /api/v1/faces/profiles returns initialized demo profiles")
    void testListProfiles() throws Exception {
        mockMvc.perform(get("/api/v1/faces/profiles"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()", greaterThanOrEqualTo(2)));
    }

    @Test
    @DisplayName("GET /api/v1/stream/snapshot returns real-time annotated JPEG bytes")
    void testGetSnapshot() throws Exception {
        mockMvc.perform(get("/api/v1/stream/snapshot"))
                .andExpect(status().isOk())
                .andExpect(result -> {
                    byte[] content = result.getResponse().getContentAsByteArray();
                    assert content.length > 0;
                });
    }
}
