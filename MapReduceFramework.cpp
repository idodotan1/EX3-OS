//
// Created by idodo on 13/05/2025.
//
#include "MapReduceFramework.h"
#include <thread>
#include <vector>
#include <cstdio>
#include <atomic>
#include <iostream>
#include <algorithm>
#include "Barrier/Barrier.h"
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



void getJobState(std::atomic<uint64_t> &state, JobState* out) {
    uint64_t snapshot = state.load();
    stage_t stage;
    uint32_t processed, total;
    unpackState(snapshot, stage, processed, total);
    out->stage = stage;
    out->percentage = (total == 0) ? 0 : (float)(100.0 * processed / total);
}
struct JobContext;
struct ThreadContext {
    int threadID;
    std::atomic<uint64_t>* atomicInputIndex{};
    IntermediateVec intermediateVec;
    JobContext* jobContext;

};
struct JobContext {
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    int multiThreadLevel;
    std::vector<std::thread> threads;
    std::vector<ThreadContext> contexts;
    Barrier barrier;
    std::atomic<uint32_t>  workIndex;
    std::atomic<uint64_t> atomicInputIndex;
    std::vector<IntermediateVec> shuffleQueue;
    std::atomic<int> shuffleCounter;
    std::mutex outputMutex;
    bool threadsJoined;
    std::mutex joinThreadsMutex;
    JobContext(const MapReduceClient& client,
                           const InputVec& inputVec,
                           OutputVec& outputVec,
                           int multiThreadLevel): client(client),
                           inputVec(inputVec), outputVec(outputVec),
              multiThreadLevel(multiThreadLevel), workIndex(0),
              barrier(multiThreadLevel), atomicInputIndex(0),
              shuffleCounter(0),
              threadsJoined(false)
    {
        threads.reserve(multiThreadLevel);
        contexts.resize(multiThreadLevel);
    }
};
void setStageAndTotal(JobContext* jobContext, stage_t stage, uint32_t total)
{
    uint64_t newVal = packState(stage, 0, total);
    jobContext->atomicInputIndex.store(newVal);
    jobContext->workIndex.store(0);
}
K2* findMaxKey(JobContext* jobContext, const std::vector<size_t>& idx) {
    K2* maxKey = nullptr;
    for (int i = 0; i < jobContext->multiThreadLevel; ++i) {
        if (idx[i] == 0) continue;
        K2* key = jobContext->contexts[i].intermediateVec[idx[i] - 1].first;
        if (!maxKey || *maxKey < *key) {
            maxKey = key;
        }
    }
    return maxKey;
}


void shuffle(JobContext* jobContext)
{
    // Back indices for each thread's intermediateVec
    std::vector<size_t> idx(jobContext->multiThreadLevel);
    for (int i = 0; i < jobContext->multiThreadLevel; ++i)
    {
        idx[i] = jobContext->contexts[i].intermediateVec.size();
    }
    while (true)
    {
        K2* maxKey = findMaxKey(jobContext,idx);
        if (!maxKey) break;  // all vectors are empty
        IntermediateVec group;
        for (int i = 0; i < jobContext->multiThreadLevel; ++i) {
            auto& vec = jobContext->contexts[i].intermediateVec;
            while (idx[i] > 0) {
                K2* currentKey = vec[idx[i] - 1].first;
                if (!(*currentKey < *maxKey) && !(*maxKey < *currentKey)) {
                    group.push_back(vec[idx[i] - 1]);
                    --idx[i];
                    jobContext->atomicInputIndex.fetch_add(1ULL << 2);
                } else {
                    break;
                }
            }
        }
        {
            jobContext->shuffleQueue.push_back(std::move(group));
        }
    }
}

void mapWorker(int threadID, ThreadContext* threadContext, JobContext* jobContext)
{
    while (true)
    {
        auto index = jobContext->workIndex.fetch_add(1);
        if (index >= jobContext->inputVec.size()) break;
        fetchNextInputIndex(&jobContext->atomicInputIndex);
        const auto& pair = jobContext->inputVec[index];
        jobContext->client.map(pair.first, pair.second, threadContext);
    }
    std::sort(threadContext->intermediateVec.begin(),
          threadContext->intermediateVec.end(),
          [](const std::pair<K2*, V2*>& a, const std::pair<K2*, V2*>& b) {
              return *a.first < *b.first;
          });
    jobContext->barrier.barrier();
    if (threadID == 0) {

        setStageAndTotal(jobContext,SHUFFLE_STAGE,
          jobContext->shuffleCounter);
        shuffle(jobContext);
        setStageAndTotal(jobContext,REDUCE_STAGE,jobContext->shuffleQueue
        .size());
    }
    jobContext->barrier.barrier();
    while (true)
    {
        auto index = jobContext->workIndex.fetch_add(1);
        if (index >= jobContext->shuffleQueue.size()) break;
        fetchNextInputIndex(&jobContext->atomicInputIndex);
        IntermediateVec group = std::move(jobContext->shuffleQueue[index]);
        jobContext->client.reduce(&group,threadContext);
    }

}



JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
    auto jobHandle = new JobContext{client,inputVec,outputVec,
                                     multiThreadLevel};
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        jobHandle->contexts[i].threadID = i;
        jobHandle->contexts[i].atomicInputIndex = &jobHandle->atomicInputIndex;
        jobHandle->contexts[i].jobContext = jobHandle;
    }
    try {
        setStageAndTotal(jobHandle, MAP_STAGE,
                         jobHandle->inputVec.size());
        for (int i = 0; i < multiThreadLevel; ++i) {
            jobHandle->threads.emplace_back(mapWorker, i,
                                            &jobHandle->contexts[i],jobHandle);
        }
    } catch (const std::system_error& e) {
        std::cerr <<  "system error: failed to create thread" << std::endl;
        delete jobHandle;
        exit(1);
    }
    return jobHandle;
}
void emit2(K2* key, V2* value, void* context) {
    auto* threadContext = static_cast<ThreadContext*>(context);
    threadContext->intermediateVec.emplace_back(key, value);
    threadContext->jobContext->shuffleCounter.fetch_add(1);
}
void emit3(K3* key, V3* value, void* context)
{
    auto* threadContext = static_cast<ThreadContext*>(context);
    std::lock_guard<std::mutex> lock(threadContext->jobContext->outputMutex);
    threadContext->jobContext->outputVec.emplace_back(key,value);
}
void waitForJob(JobHandle job) {
    auto jobHandle = static_cast<JobContext*>(job);
    std::lock_guard<std::mutex> lock(jobHandle->joinThreadsMutex);
    if (!jobHandle->threadsJoined) {
        jobHandle->threadsJoined = true;
        for (std::thread& t : jobHandle->threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

}
void closeJobHandle(JobHandle job) {
    waitForJob(job);
    delete static_cast<JobContext*>(job);
}
void getJobState(JobHandle job, JobState* state) {
    auto jobHandle = static_cast<JobContext*>(job);
    getJobState(jobHandle->atomicInputIndex, state);
}