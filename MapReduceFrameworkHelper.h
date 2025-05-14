//
// Created by idodo on 13/05/2025.
//

#ifndef EX3_MAPREDUCEFRAMEWORKHELPER_H
#define EX3_MAPREDUCEFRAMEWORKHELPER_H
#include "MapReduceClient.h"
#include <atomic>
#include <vector>

struct ThreadContext;
struct JobContext;
void mapWorker(int threadID, ThreadContext* threadContext, JobContext* jobContext);

#endif //EX3_MAPREDUCEFRAMEWORKHELPER_H
