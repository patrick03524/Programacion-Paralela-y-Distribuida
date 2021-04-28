/* Instructions 
 * File: pth_rwl.c
 * Compile: gcc -g -Wall -o pth_rwl pth_rwl.c -lpthread
 * Usage: ./pth_rwl <thread_count>
 * Section 4.9.3
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "rand.h"
#include <pthread.h>

const int MAX_KEY = 100000000;
int act_keys = 100;

struct list_node_s {
  int data;
  struct list_node_s* next;
  pthread_mutex_t mutex;
};

struct list_node_s* head = NULL;
int thread_count;
int total_ops;
double insert_p;
double search_p;
double delete_p;
pthread_rwlock_t rwlock;
pthread_mutex_t count_mutex;
int Member_count = 0, insert_count = 0, delete_count = 0;

/* Setup */
void Usage(char* prog_name);
void Get_input(int* inserts_in_main_p);

/* Thread function */
void* Thread_work(void* rank);

/* List operations */
int Insert(int value);
void Print(void);
int Member(int value);
int Delete(int value);
void Free_list(void);
int Is_empty(void);

int main(int argc, char* argv[]) {
   long i; 
   int key, success, attempts;
   pthread_t* thread_handles;
   int inserts_in_main;
   unsigned seed = 1;
   double start, finish;
   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1],NULL,10);
   Get_input(&inserts_in_main);
   i = attempts = 0;
   while ( i < inserts_in_main && attempts < 2*inserts_in_main ) {
      key = my_rand(&seed) % MAX_KEY;
      success = Insert(key);
      attempts++;
      if (success) i++;
   }
   printf("Inserted %ld keys in empty list\n", i);
   thread_handles = malloc(thread_count*sizeof(pthread_t));
   pthread_mutex_init(&count_mutex, NULL);
   pthread_rwlock_init(&rwlock, NULL);
   GET_TIME(start);
   for (i = 0; i < thread_count; i++)
      pthread_create(&thread_handles[i], NULL, Thread_work, (void*) i);

   for (i = 0; i < thread_count; i++)
      pthread_join(thread_handles[i], NULL);
   GET_TIME(finish);
   Print();
   printf("Elapsed time = %e seconds\n", finish - start);
   printf("Total ops = %d\n", total_ops);
   printf("Member(FIND) ops = %d\n", Member_count);
   printf("insert ops = %d\n", insert_count);
   printf("delete ops = %d\n", delete_count);
   /* Free List and Threads */
   Free_list();
   pthread_rwlock_destroy(&rwlock);
   pthread_mutex_destroy(&count_mutex);
   free(thread_handles);
   return 0;
}  /* main */

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count>\n", prog_name);
   exit(0);	
}  /* Usage */

void Get_input(int* inserts_in_main_p) {
   printf("Keys: \n");
   scanf("%d", inserts_in_main_p);
   printf("Ops Number: \n");
   scanf("%d", &total_ops);
   printf("Percent of Ops in searches? (between 0 and 1)\n");
   scanf("%lf", &search_p);
   printf("Percent of Ops in inserts? (between 0 and 1)\n");
   scanf("%lf", &insert_p);
   delete_p = 1.0 - (search_p + insert_p);
}  /* Get_input */

int Insert(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   struct list_node_s* temp;
   int rv = 1;
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }
   if (curr == NULL || curr->data > value) {
      temp = malloc(sizeof(struct list_node_s));
      temp->data = value;
      temp->next = curr;
      if (pred == NULL)
         head = temp;
      else
         pred->next = temp;
   } else { /* value in list */
      rv = 0;
   }

   return rv;
}  /* Insert */

void Print(void) {
   struct list_node_s* temp;
   printf("lista = ");
   temp = head;
   while (temp != (struct list_node_s*) NULL) {
      printf("%d ", temp->data);
      temp = temp->next;
   }
   printf("\n");
}  /* Print */

int  Member(int value) {
   struct list_node_s* temp;
   temp = head;
   while (temp != NULL && temp->data < value)
      temp = temp->next;
   if (temp == NULL || temp->data > value) {
      return 0;
   } else {
      return 1;
   }
}  /* Member */

int Delete(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   int rv = 1;
   /* Member value */
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }
   if (curr != NULL && curr->data == value) {
      if (pred == NULL) { /* first element in list */
         head = curr->next;
         free(curr);
      } else { 
         pred->next = curr->next;
         free(curr);
      }
   } else { /* Not in list */
      rv = 0;
   }

   return rv;
}  /* Delete */

void Free_list(void) {
   struct list_node_s* p;
   struct list_node_s* temp;
   if (Is_empty()) return;
   p = head; 
   temp = p->next;
   while (temp != NULL) {
      free(p);
      p = temp;
      temp = p->next;
   }
   free(p);
}  /* Free_list */

int  Is_empty(void) {
   if (head == NULL)
      return 1;
   else
      return 0;
}  /* Is_empty */

void* Thread_work(void* rank) {
   long my_rank = (long) rank;
   int i, val;
   double which_op;
   unsigned seed = my_rank + 1;
   int my_Member_count = 0, my_insert_count=0, my_delete_count=0;
   int ops_per_thread = total_ops/thread_count;
   for (i = 0; i < ops_per_thread; i++) {
      which_op = my_drand(&seed);
      val = my_rand(&seed) % MAX_KEY;
      if (which_op < search_p) {
         pthread_rwlock_rdlock(&rwlock);
         Member(val);
         pthread_rwlock_unlock(&rwlock);
         my_Member_count++;
      } else if (which_op < search_p + insert_p) {
         pthread_rwlock_wrlock(&rwlock);
         Insert(val);
         pthread_rwlock_unlock(&rwlock);
         my_insert_count++;
      } else { /* delete */
         pthread_rwlock_wrlock(&rwlock);
         Delete(val);
         pthread_rwlock_unlock(&rwlock);
         my_delete_count++;
      }
   }  /* for */
   pthread_mutex_lock(&count_mutex);
   Member_count += my_Member_count;
   insert_count += my_insert_count;
   delete_count += my_delete_count;
   pthread_mutex_unlock(&count_mutex);
   return NULL;
}  /* Thread_work */