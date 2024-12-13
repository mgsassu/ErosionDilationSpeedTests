#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool
{
public:
    // Constructor: Initializes the thread pool with a given number of threads
    explicit ThreadPool(int numThreads);

    // Destructor: Waits for all threads to finish and cleans up
    ~ThreadPool();

    // Enqueues a task into the thread pool
    void enqueueTask(std::function<void()> task);

private:
    std::vector<std::thread> workers;              // Vector of worker threads
    std::queue<std::function<void()>> tasks;       // Queue of tasks
    std::mutex queueMutex;                         // Mutex for thread-safe task queue
    std::condition_variable condition;             // Condition variable for task synchronization
    bool stop = false;
};

