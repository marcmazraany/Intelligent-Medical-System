package com.intellimeds.api.medications;

import com.intellimeds.api.common.NotFoundException;
import com.intellimeds.api.medications.dto.MedicationDto;
import com.intellimeds.api.medications.dto.UpsertMedicationRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.UUID;

@Service
public class MedicationService {

    private final MedicationRepository repo;

    public MedicationService(MedicationRepository repo) {
        this.repo = repo;
    }

    @Transactional(readOnly = true)
    public List<MedicationDto> list(UUID userId) {
        return repo.findAllByUserIdOrderByUpdatedAtDesc(userId)
                .stream()
                .map(MedicationService::toDto)
                .toList();
    }

    @Transactional
    public MedicationDto create(UUID userId, UpsertMedicationRequest req) {
        MedicationEntity e = MedicationEntity.builder()
                .userId(userId)
                .name(req.name())
                .dosage(req.dosage())
                .expiryDate(req.expiryDate())
                .frequency(req.frequency())
                .quantity(normalizeQuantity(req.quantity()))
                .reminderTimes(normalizeReminderTimes(req.reminderTimes()))
                .status(parseStatus(req.status()))
                .notes(req.notes())
                .build();
        return toDto(repo.save(e));
    }

    @Transactional
    public MedicationDto update(UUID userId, UUID id, UpsertMedicationRequest req) {
        MedicationEntity e = repo.findByIdAndUserId(id, userId)
                .orElseThrow(() -> new NotFoundException("Medication not found"));

        e.setName(req.name());
        e.setDosage(req.dosage());
        e.setExpiryDate(req.expiryDate());
        e.setFrequency(req.frequency());
        e.setQuantity(normalizeQuantity(req.quantity()));
        e.setReminderTimes(normalizeReminderTimes(req.reminderTimes()));
        e.setStatus(parseStatus(req.status()));
        e.setNotes(req.notes());

        return toDto(repo.save(e));
    }

    @Transactional
    public void delete(UUID userId, UUID id) {
        repo.findByIdAndUserId(id, userId)
                .orElseThrow(() -> new NotFoundException("Medication not found"));
        repo.deleteByIdAndUserId(id, userId);
    }

    private static Integer normalizeQuantity(Integer quantity) {
        return (quantity == null || quantity < 1) ? 1 : quantity;
    }

    private static List<String> normalizeReminderTimes(List<String> reminderTimes) {
        if (reminderTimes == null || reminderTimes.isEmpty()) return List.of();

        DateTimeFormatter input = new DateTimeFormatterBuilder()
                .appendPattern("H:mm")
                .toFormatter();
        DateTimeFormatter out = DateTimeFormatter.ofPattern("HH:mm");

        var seen = new LinkedHashSet<String>();

        for (String t : reminderTimes) {
            if (t == null) continue;
            String trimmed = t.trim();
            if (trimmed.isEmpty()) continue;

            try {
                LocalTime lt = LocalTime.parse(trimmed, input);
                seen.add(lt.format(out));
            } catch (DateTimeParseException ex) {
                throw new IllegalArgumentException("Invalid reminderTime: " + trimmed + " (expected HH:mm)");
            }
        }

        return new ArrayList<>(seen);
    }

    private static MedicationStatus parseStatus(String status) {
        return switch (status) {
            case "available" -> MedicationStatus.available;
            case "low-stock", "low_stock" -> MedicationStatus.low_stock;
            case "expiring-soon", "expiring_soon" -> MedicationStatus.expiring_soon;
            default -> MedicationStatus.available;
        };
    }

    private static MedicationDto toDto(MedicationEntity e) {
        return MedicationDto.builder()
                .id(e.getId().toString())
                .name(e.getName())
                .dosage(e.getDosage())
                .expiryDate(e.getExpiryDate())
                .frequency(e.getFrequency())
                .quantity(e.getQuantity())
                .reminderTimes(e.getReminderTimes())
                .status(e.getStatus().name())
                .notes(e.getNotes())
                .build();
    }
}