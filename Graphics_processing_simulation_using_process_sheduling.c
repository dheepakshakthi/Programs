/*
GRAPHICS PROCESSING SIMULATION USING PROCESS SCHEDULING WITH RESOURCE ALLOCATION

The primary objective is to implement a scheduling system that integrates priority based resource allocation in process scheduling for graphics processing. The objective includes: 
1.	Process scheduling: The program aims to simulate a process scheduler that manages multiple processes, each with specific resource requirements.
2.	Dynamic Resource Allocation: The scheduler ensures that processes are allocated with necessary resources such as memory and I/O units for execution and prevents deadlock to improve system performance.
3.	Concurrency: The program uses multithreading to handle the scheduling of processes concurrently. 
4.	Synchronization: The program uses synchronization mechanisms to manage the waiting and retrying of processes when resources are insufficient. 
5.	Process prioritization: The scheduler likely prioritizes processes based on certain criteria to determine the order of execution.
6.	Simulation: The program simulates the behaviour of an operating system’s process scheduler. providing insights into how processes are managed and scheduled in a real-world scenario.
*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>

#define NAME_WIDTH 20

const char* process_names[] = {"Render", "Texture", "Shader", "Lighting", "Physics"};

typedef struct {
    char name[50];
    int id;
    int cpu_burst;
    int memory;
    int io;
    int priority;
    int waiting_time;
    int turnaround_time;
    bool executed;
} Process;

typedef struct {
    Process* processes;
    int size;
    int capacity;
    int available_memory;
    int available_io;
    pthread_mutex_t mtx;
} Scheduler;

void initScheduler(Scheduler* scheduler, int mem, int io, int capacity) {
    scheduler->processes = (Process*)malloc(sizeof(Process) * capacity);
    scheduler->size = 0;
    scheduler->capacity = capacity;
    scheduler->available_memory = mem;
    scheduler->available_io = io;
    pthread_mutex_init(&scheduler->mtx, NULL);
}

void addProcess(Scheduler* scheduler, int id, int cpu_burst, int memory, int io, int priority) {
    pthread_mutex_lock(&scheduler->mtx);
    if (scheduler->size < scheduler->capacity) {
        strncpy(scheduler->processes[scheduler->size].name, process_names[id], sizeof(scheduler->processes[scheduler->size].name) - 1);
        scheduler->processes[scheduler->size].name[sizeof(scheduler->processes[scheduler->size].name) - 1] = '\0';
        scheduler->processes[scheduler->size].id = id;
        scheduler->processes[scheduler->size].cpu_burst = cpu_burst;
        scheduler->processes[scheduler->size].memory = memory;
        scheduler->processes[scheduler->size].io = io;
        scheduler->processes[scheduler->size].priority = priority;
        scheduler->processes[scheduler->size].waiting_time = 0;
        scheduler->processes[scheduler->size].turnaround_time = 0;
        scheduler->processes[scheduler->size].executed = false;
        scheduler->size++;
    }
    pthread_mutex_unlock(&scheduler->mtx);
}

void executeProcess(Process* p, int* current_time) {
    printf("Executing Process %-*s (ID: %d)\n", NAME_WIDTH, p->name, p->id);
    sleep(p->cpu_burst);
    *current_time += p->cpu_burst;
    p->turnaround_time = *current_time;
    p->executed = true;
    printf("Process %-*s executed successfully.\n", NAME_WIDTH, p->name);
}

void* scheduleProcesses(void* arg) {
    Scheduler* scheduler = (Scheduler*)arg;
    int current_time = 0;
    
    while (true) {
        pthread_mutex_lock(&scheduler->mtx);

        if (scheduler->size == 0) {
            pthread_mutex_unlock(&scheduler->mtx);
            break;
        }

        Process* best_process = NULL;
        int index = -1;

        for (int i = 0; i < scheduler->size; ++i) {
            if (!scheduler->processes[i].executed &&
                scheduler->processes[i].memory <= scheduler->available_memory &&
                scheduler->processes[i].io <= scheduler->available_io) {
                if (!best_process || scheduler->processes[i].priority < best_process->priority) {
                    best_process = &scheduler->processes[i];
                    index = i;
                }
            }
        }

        if (best_process) {
            scheduler->available_memory -= best_process->memory;
            scheduler->available_io -= best_process->io;
            best_process->waiting_time = current_time;
            pthread_mutex_unlock(&scheduler->mtx);

            executeProcess(best_process, &current_time);

            pthread_mutex_lock(&scheduler->mtx);
            scheduler->available_memory += best_process->memory;
            scheduler->available_io += best_process->io;
            scheduler->processes[index].executed = true;
            pthread_mutex_unlock(&scheduler->mtx);
        } else {
            printf("\nNo process can be executed due to insufficient resources.\n");
            pthread_mutex_unlock(&scheduler->mtx);
            break;
        }
    }
    return NULL;
}

void printExecutionSummary(Scheduler* scheduler) {
    printf("\n%-*s %-10s %-10s %-10s %s\n", NAME_WIDTH, "Name", "Waiting", "Turnaround", "Executed", "Status");
    printf("%-*s %-10s %-10s %-10s %s\n", NAME_WIDTH, "--------------------", "----------", "----------", "----------", "---------");
    for (int i = 0; i < scheduler->size; i++) {
        printf("%-*s %-10d %-10d %-10s %s\n", NAME_WIDTH, scheduler->processes[i].name, scheduler->processes[i].waiting_time,
               scheduler->processes[i].turnaround_time, scheduler->processes[i].executed ? "✔" : "✘",
               scheduler->processes[i].executed ? "Completed" : "Not Executed");
    }
    printf("%-*s %-10s %-10s %-10s %s\n", NAME_WIDTH, "--------------------", "----------", "----------", "----------", "---------");
}

int main() {
    Scheduler scheduler;
    initScheduler(&scheduler, 100, 50, 5);

    for (int i = 0; i < 5; i++) {
        int cpu_burst, memory, io, priority;
        printf("Enter details for %s (CPU_Burst Memory IO Priority): ", process_names[i]);
        scanf("%d %d %d %d", &cpu_burst, &memory, &io, &priority);
        addProcess(&scheduler, i, cpu_burst, memory, io, priority);
    }
    
    pthread_t scheduler_thread;
    pthread_create(&scheduler_thread, NULL, scheduleProcesses, &scheduler);
    pthread_join(scheduler_thread, NULL);
    
    printf("\nExecution Summary:\n");
    printExecutionSummary(&scheduler);

    printf("\nScheduler execution completed. Exiting program.\n");

    free(scheduler.processes);
    pthread_mutex_destroy(&scheduler.mtx);
    return 0;
}
