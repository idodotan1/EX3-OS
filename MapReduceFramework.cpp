//
// Created by idodo on 13/05/2025.
//
#include "MapReduceFramework.h"
#include "MapReduceFrameworkHelper.h"
#include <thread>
#include <vector>
#include <cstdio>
#include <atomic>
#include <iostream>
#include "Barrier.h"
// Bit masks
constexpr uint64_t STAGE_MASK = 0x3ULL;
constexpr uint64_t PROCESSED_MASK = 0x7FFFFFFFULL << 2;
constexpr uint64_t TOTAL_MASK = 0x7FFFFFFFULL << 33;

uint64_t packState(stage_t stage, uint32_t processed, uint32_t total) {
    return ((uint64_t)stage) |
           ((uint64_t)processed << 2) |
           ((uint64_t)total << 33);
}

void unpackState(uint64_t packed, stage_t &stage, uint32_t &processed, uint32_t &total) {
    stage = static_cast<stage_t>(packed & STAGE_MASK);
    processed = (packed >> 2) & 0x7FFFFFFF;
    total = (packed >> 33) & 0x7FFFFFFF;
}

uint32_t fetchNextInputIndex(std::atomic<uint64_t>* state) {
    return ((state->fetch_add(1ULL << 2) >> 2) & 0x7FFFFFFF);
}

void setStage(std::atomic<uint64_t> &state, stage_t newStage) {
    uint64_t old = state.load();
    while (true) {
        uint64_t newVal = (old & ~STAGE_MASK) | ((uint64_t)newStage);
        if (state.compare_exchange_weak(old, newVal)) break;
    }
}

void setTotal(std::atomic<uint64_t> &state, uint32_t total) {
    uint64_t old = state.load();
    while (true) {
        uint64_t newVal = (old & STAGE_MASK) | ((uint64_t)total << 33);
        if (state.compare_exchange_weak(old, newVal)) break;
    }
}

void getJobState(std::atomic<uint64_t> &state, JobState* out) {
    uint64_t snapshot = state.load();
    stage_t stage;
    uint32_t processed, total;
    unpackState(snapshot, stage, processed, total);
    out->stage = stage;
    out->percentage = (total == 0) ? 0 : (float)(100.0 * processed / total);
}

struct ThreadContext {
    int threadID;
    std::atomic<uint64_t>* atomicInputIndex{};
    IntermediateVec intermediateVec;
};
struct JobContext {
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    int multiThreadLevel;
    std::vector<std::thread> threads;
    std::vector<ThreadContext> contexts;
    Barrier barrier;
    std::atomic<uint64_t> atomicInputIndex;
    std::vector<IntermediateVec> shuffleQueue;
    std::atomic<int> shuffleCounter;
    JobState jobState;
    std::mutex stateMutex;
    std::mutex outputMutex;
    std::mutex shuffleMutex;
    std::atomic<bool> threadsJoined;
    JobContext(const MapReduceClient& client,
                           const InputVec& inputVec,
                           OutputVec& outputVec,
                           int multiThreadLevel): client(client),
                           inputVec(inputVec), outputVec(outputVec),
              multiThreadLevel(multiThreadLevel),
              barrier(multiThreadLevel), atomicInputIndex(0),
              shuffleCounter(0),
              jobState{UNDEFINED_STAGE, 0},
              threadsJoined(false)
    {
        threads.reserve(multiThreadLevel);
        contexts.reserve(multiThreadLevel);
    }
};

void mapWorker(int threadID, ThreadContext* threadContext, JobContext* jobContext)
{
    while (true) {
        auto index = fetchNextInputIndex(threadContext->atomicInputIndex);
        if (index >= jobContext->inputVec.size()) break;
        const auto& pair = jobContext->inputVec[index];
        jobContext->client.map(pair.first, pair.second, threadContext);
    }
}



JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
    auto jobHandle = new JobContext{client,inputVec,outputVec,
                                     multiThreadLevel};
    for (int i = 0; i < multiThreadLevel; ++i) {
        jobHandle->contexts[i].threadID = i;
        jobHandle->contexts[i].atomicInputIndex = &jobHandle->atomicInputIndex;    }
    try {
        for (int i = 0; i < multiThreadLevel; ++i) {
            jobHandle->threads.emplace_back(mapWorker, i,
                                            &jobHandle->contexts[i],jobHandle);
        }
    } catch (const std::system_error& e) {
        std::cerr <<  "system error: failed to create thread" << std::endl;
        delete jobHandle;
        exit(1);
    }
}