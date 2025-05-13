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
                           int multiThreadLevel)
            : client(client),
              inputVec(inputVec),
              outputVec(outputVec),
              multiThreadLevel(multiThreadLevel),
              barrier(multiThreadLevel),
              atomicInputIndex(0),
              shuffleCounter(0),
              jobState{UNDEFINED_STAGE, 0},
              threadsJoined(false)
    {
        threads.reserve(multiThreadLevel);
        contexts.reserve(multiThreadLevel);
    }
};

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
            jobHandle->threads.emplace_back(mapWorker, i, &jobHandle->contexts[i]);
        }
    } catch (const std::system_error& e) {
        std::cerr <<  "system error: failed to create thread" << std::endl;
        delete jobHandle;
        exit(1);
    }
}