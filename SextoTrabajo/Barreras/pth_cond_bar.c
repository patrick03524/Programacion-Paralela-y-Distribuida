/* Instructions 
 * File: pth_cond_bar.c
 * Compile: gcc -g -Wall -o pth_cond_bar pth_cond_bar.c -lpthread
 * Usage: ./pth_cond_bar <thread_count>
 * Section 4.8.3 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "timer.h"

#define BARRIER_COUNT 100

int thread_count;
int barrier_thread_count = 0;
pthread_mutex_t barrier_mutex;
pthread_cond_t ok_to_proceed;

void Usage(char* prog_name); /* Funtion that ask the number of threads */
void *Thread_work(void* rank); /* Work of every Thread */

int main(int argc, char* argv[]) {
   long       thread;
   pthread_t* thread_handles; 
   double start, finish;
   if (argc != 2)
      Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   thread_handles = malloc (thread_count*sizeof(pthread_t));
   pthread_mutex_init(&barrier_mutex, NULL);
   pthread_cond_init(&ok_to_proceed, NULL);
   GET_TIME(start);
   /* Initialize every pthread */
   for (thread = 0; thread < thread_count; thread++)
      pthread_create(&thread_handles[thread], NULL,
          Thread_work, (void*) thread);

   for (thread = 0; thread < thread_count; thread++) {
      pthread_join(thread_handles[thread], NULL);
   }
   /* Work Done*/
   GET_TIME(finish);
   /* GET TIME */
   printf("Elapsed time = %e seconds\n", finish - start);
   /* FREE PTHREADS */
   pthread_mutex_destroy(&barrier_mutex);
   pthread_cond_destroy(&ok_to_proceed);
   free(thread_handles);
   return 0;
}  /* main */

void Usage(char* prog_name) {

   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   exit(0);
}  /* Usage */

void *Thread_work(void* rank) {
   for (int i = 0; i < BARRIER_COUNT; i++) {
      /* Barrier: 4.8.3 */
      pthread_mutex_lock(&barrier_mutex);
      barrier_thread_count++;
      if (barrier_thread_count == thread_count) {
         barrier_thread_count = 0;
         pthread_cond_broadcast(&ok_to_proceed);
      } else {
         while (pthread_cond_wait(&ok_to_proceed,
                   &barrier_mutex) != 0);
      }
      pthread_mutex_unlock(&barrier_mutex);
   }
   return NULL;
}  /* Thread_work */