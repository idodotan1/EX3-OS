//
// Created by idodo on 13/05/2025.
//

#include "MapReduceFrameworkHelper.h"
void mapWorker(int threadID, ThreadContext* threadContext, JobContext* jobContext)
{
    while (true) {
        int index = threadContext->atomicInputIndex->fetch_add(1);
        if (index >= jobContext->inputVec.size()) break;

        const auto& pair = jobContext->inputVec[index];
        jobContext->client.map(pair.first, pair.second, threadContext);
    }
}
